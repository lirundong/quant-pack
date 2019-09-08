# -*- coding: utf-8 -*-

import math
from copy import copy
from types import MethodType
from logging import getLogger
from functools import wraps
from contextlib import contextmanager

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.nn.modules._functions import SyncBatchNorm as sync_batch_norm

if torch.cuda.is_available():
    from quant_prob.modeling.quantizers.cuda_param_linear_quantizer import cuda_fake_linear_quant
    quantizer = cuda_fake_linear_quant
else:
    from quant_prob.modeling.quantizers.param_linear_quantizer import fake_linear_quant
    quantizer = fake_linear_quant

__all__ = ["IDQ"]


class IDQ:

    def __init__(self, forward_func, kw=4, ka=4, fp_layers=None, align_zero=True,
                 devices=None, non_blocking=False):
        assert isinstance(self, nn.Module), f"IDQ should be used in conjunction with `nn.Module`"

        if fp_layers is not None and not isinstance(fp_layers, (list, tuple)):
            fp_layers = (fp_layers, )

        self.kw = kw
        self.ka = ka
        self.fp_layers = fp_layers
        self.align_zero = align_zero
        self.weight_quant_param = nn.ParameterDict()
        self.activation_quant_param = nn.ParameterDict()
        self.layer_names = dict()

        # devices for model-parallel
        if devices is None:
            if torch.cuda.is_available():
                self.fp_device = self.q_device = torch.device(f"cuda:{torch.cuda.current_device()}")
            else:
                self.fp_device = self.q_device = torch.device("cpu")
        else:
            assert len(devices) == 2
            self.fp_device = devices[0]
            self.q_device = devices[1]
        self.non_blocking = non_blocking

        # Since ordinary hooks do not get invoked when module being wrapped by
        # DDP, we re-initialize them in `forward` at specific iterations.
        # Currently we only use `forward_hooks`.
        self.ddp_forward_hooks = []
        self.ddp_hook_handles = []

        # TODO: a little bit tricky, can we make this more elegant?
        @self._wrap_forward()
        @wraps(forward_func)
        def _do_forward(obj, x):
            return forward_func(obj, x)
        # do not need `MethodType` here, because self has already bind to `_do_forward`
        self.forward = _do_forward

        self._init_quant_param()
        self.to_proper_device()

    def _init_quant_param(self):
        for n, m in self.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                self.layer_names[id(m)] = n
                lb = Parameter(m.weight.detach().min())
                ub = Parameter(m.weight.detach().max())
                self.weight_quant_param[f"{n}_weight_lb".replace(".", "_")] = lb
                self.weight_quant_param[f"{n}_weight_ub".replace(".", "_")] = ub
                self.activation_quant_param[f"{n}_act_lb".replace(".", "_")] = Parameter(torch.tensor(0.))
                self.activation_quant_param[f"{n}_act_ub".replace(".", "_")] = Parameter(torch.tensor(0.))

    def _update_stat(self, input, name, percentile, param_bank):
        assert torch.is_tensor(input)
        assert not input.requires_grad
        if percentile == 1.:
            v = input.max()
        elif percentile == 0.:
            v = input.min()
        else:
            assert input.dim() == 1
            k = int(math.floor(input.numel() * percentile))
            v = input[k]
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.all_reduce(v)
            v.div_(dist.get_world_size())
        try:
            param_bank[name].detach_().copy_(v, non_blocking=self.non_blocking)
        except KeyError as e:
            raise RuntimeError(f"update quant-param which not seen in init: `{e}`")

    def _update_weight_quant_param(self):
        for n, m in self.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                weight_view = m.weight.detach().reshape(-1)
                self._update_stat(weight_view, f"{n}_weight_lb".replace(".", "_"), 0., self.weight_quant_param)
                self._update_stat(weight_view, f"{n}_weight_ub".replace(".", "_"), 1., self.weight_quant_param)

    def _prepare_calibration_hooks(self, gamma=0.999, update_bn=False):

        def update_act_stat_hook(module, input, output):
            if not isinstance(module, (nn.Conv2d, nn.Linear)):
                return

            name = self.layer_names[id(module)]
            if not torch.is_tensor(input):
                assert len(input) == 1
                input = input[0]
            if input.requires_grad:
                input = input.detach()
            input_view, _ = input.reshape(-1).sort()
            input_view = input_view.to(self.fp_device, non_blocking=self.non_blocking)
            self._update_stat(input_view, f"{name}_act_ub".replace(".", "_"), gamma, self.activation_quant_param)
            self._update_stat(input_view, f"{name}_act_lb".replace(".", "_"), 1. - gamma, self.activation_quant_param)

        def update_bn_stat_hook(module, input, output):
            if not isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                return

            if not torch.is_tensor(input):
                assert len(input) == 1
                input = input[0]
            if input.requires_grad:
                input = input.detach()
            assert input.dim() == 4
            n, c = input.shape[:2]
            input_view = input.permute(1, 0, 2, 3).reshape(c, -1)
            if isinstance(module, nn.SyncBatchNorm):
                c_sum = input_view.sum(dim=1)
                c_square_sum = input_view.pow(2).sum(dim=1)
                n = torch.tensor(n, device=input.device, dtype=input.dtype)
                dist.all_reduce(c_sum)
                dist.all_reduce(c_square_sum)
                dist.all_reduce(n)
                mean = c_sum / n
                var = c_square_sum / n - mean.pow(2)  # TODO: unbiased?
            else:
                mean = input_view.mean()
                var = input_view.var()

            module.running_mean.copy_(mean)
            module.running_var.copy_(var)

        assert len(self.ddp_forward_hooks) == 0
        self.ddp_forward_hooks.append(update_act_stat_hook)
        if update_bn:
            self.ddp_forward_hooks.append(update_bn_stat_hook)

    def _wrap_forward(self):
        # TODO: refactor `prepare_quant_forward` to a context manager and move
        #       it out of this wrapper

        def do_fake_quant(name, weight, x):
            # TODO: looks ugly, can we find something like context manager to
            #       do the device mapping?
            assert weight.device == x.device == self.q_device
            w_lb = self.weight_quant_param[f"{name}_weight_lb".replace(".", "_")]
            w_ub = self.weight_quant_param[f"{name}_weight_ub".replace(".", "_")]
            x_lb = self.activation_quant_param[f"{name}_act_lb".replace(".", "_")]
            x_ub = self.activation_quant_param[f"{name}_act_ub".replace(".", "_")]
            w_lb = w_lb.to(self.q_device, non_blocking=self.non_blocking)
            w_ub = w_ub.to(self.q_device, non_blocking=self.non_blocking)
            x_lb = x_lb.to(self.q_device, non_blocking=self.non_blocking)
            x_ub = x_ub.to(self.q_device, non_blocking=self.non_blocking)
            qw = quantizer(weight, w_lb, w_ub, self.kw, self.align_zero)
            qx = quantizer(x, x_lb, x_ub, self.ka, self.align_zero)
            return qx, qw

        def quant_conv2d_forward(m, x):
            assert isinstance(m, nn.Conv2d)
            name = self.layer_names[id(m)]
            x = x.to(self.q_device, non_blocking=self.non_blocking)
            w = m.weight.to(self.q_device, non_blocking=self.non_blocking)
            bias = m.bias.to(self.q_device, non_blocking=self.non_blocking) if m.bias is not None else None
            qx, qw = do_fake_quant(name, w, x)
            return F.conv2d(qx, qw, bias, m.stride, m.padding, m.dilation, m.groups)

        def quant_linear_forward(m, x):
            assert isinstance(m, nn.Linear)
            name = self.layer_names[id(m)]
            x = x.to(self.q_device, non_blocking=self.non_blocking)
            w = m.weight.to(self.q_device, non_blocking=self.non_blocking)
            bias = m.bias.to(self.q_device, non_blocking=self.non_blocking) if m.bias is not None else None
            qx, qw = do_fake_quant(name, w, x)
            return F.linear(qx, qw, bias)

        def _check_bn_input_dim(x):
            if x.dim() <= 2:
                raise ValueError('expected at least 3D input (got {}D input)'.format(x.dim()))

        @contextmanager
        def _copy_bn_parameters(m):
            q_running_mean = m.running_mean.to(self.q_device, non_blocking=self.non_blocking)
            q_running_var = m.running_var.to(self.q_device, non_blocking=self.non_blocking)
            if m.affine:
                weight = m.weight.to(self.q_device, non_blocking=self.non_blocking)
                bias = m.bias.to(self.q_device, non_blocking=self.non_blocking)
            else:
                weight = bias = None

            try:
                yield q_running_mean, q_running_var, weight, bias
            finally:
                m.running_mean.copy_(q_running_mean)
                m.running_var.copy_(q_running_var)

        def bn_forward(m, x):
            # from _BatchNorm in torch-1.2
            _check_bn_input_dim(x)

            # exponential_average_factor is m.momentum set to
            # (when it is available) only so that if gets updated
            # in ONNX graph when this node is exported to ONNX.
            if m.momentum is None:
                exponential_average_factor = 0.0
            else:
                exponential_average_factor = m.momentum

            if m.training and m.track_running_stats:
                # TODO: if statement only here to tell the jit to skip emitting this when it is None
                if m.num_batches_tracked is not None:
                    m.num_batches_tracked += 1
                    if m.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(m.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = m.momentum

            with _copy_bn_parameters(m) as (q_mean, q_var, w, b):
                x_normed = F.batch_norm(
                    x, q_mean, q_var, w, b,
                    m.training or not m.track_running_stats,
                    exponential_average_factor, m.eps)
            return x_normed

        def sync_bn_forward(m, x):
            assert isinstance(m, nn.SyncBatchNorm)
            # from SyncBatchNorm in torch-1.2
            # currently only GPU input is supported
            if not x.is_cuda:
                raise ValueError('expected x tensor to be on GPU')

            if not m.ddp_gpu_size:
                raise AttributeError('SyncBatchNorm is only supported within torch.nn.parallel.DistributedDataParallel')

            _check_bn_input_dim(x)

            exponential_average_factor = 0.0

            if m.training and m.track_running_stats:
                m.num_batches_tracked += 1
                if m.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / m.num_batches_tracked.item()
                else:  # use exponential moving average
                    exponential_average_factor = m.momentum

            world_size = 1
            process_group = torch.distributed.group.WORLD
            if m.process_group:
                process_group = m.process_group
            world_size = torch.distributed.get_world_size(process_group)

            with _copy_bn_parameters(m) as (q_mean, q_var, w, b):
                # fallback to framework BN when synchronization is not necessary
                if world_size == 1 or (not m.training and m.track_running_stats):
                    normed_x = F.batch_norm(
                        x, q_mean, q_var, w, b,
                        m.training or not m.track_running_stats,
                        exponential_average_factor, m.eps)
                else:
                    normed_x = sync_batch_norm.apply(
                        x, w, b, q_mean, q_var,
                        m.eps, exponential_average_factor, process_group, world_size)
            return normed_x

        def prepare_quant_forward():
            for n, m in self.named_modules():
                if self.fp_layers is not None and any(fp_n in n for fp_n in self.fp_layers):
                    continue
                if isinstance(m, nn.Conv2d):
                    m.forward = MethodType(quant_conv2d_forward, m)
                elif isinstance(m, nn.Linear):
                    m.forward = MethodType(quant_linear_forward, m)
                elif isinstance(m, nn.BatchNorm2d):
                    m.forward = MethodType(bn_forward, m)
                elif isinstance(m, nn.SyncBatchNorm):
                    m.forward = MethodType(sync_bn_forward, m)

        def recover_fp_forward():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m.forward = MethodType(nn.Conv2d.forward, m)
                elif isinstance(m, nn.Linear):
                    m.forward = MethodType(nn.Linear.forward, m)
                elif isinstance(m, nn.BatchNorm2d):
                    m.forward = MethodType(nn.BatchNorm2d.forward, m)
                elif isinstance(m, nn.SyncBatchNorm):
                    m.forward = MethodType(nn.SyncBatchNorm.forward, m)

        def map_input_to_proper_device(*args, is_quant, **kwargs):
            device = self.q_device if is_quant else self.fp_device
            dev_args = []
            dev_kwargs = {}
            for arg in args:
                if torch.is_tensor(arg):
                    dev_args.append(arg.to(device, non_blocking=True))
                else:
                    dev_args.append(arg)
            for k, v in kwargs:
                if torch.is_tensor(v):
                    dev_kwargs[k] = v.to(device, non_blocking=True)
                else:
                    dev_kwargs[k] = v
            return dev_args, dev_kwargs

        def _decorate(func):
            @wraps(func)
            def _wrapper(*args,
                         enable_fp=True,
                         enable_quant=True,
                         update_quant_param=False,
                         update_bn=False,
                         **kwargs):

                if update_quant_param:
                    assert len(self.ddp_forward_hooks) > 0 and len(self.ddp_hook_handles) == 0
                    assert enable_fp
                    assert not enable_quant
                    ddp_forward_hooks = set(self.ddp_forward_hooks)
                    for n, m in self.named_modules():
                        if isinstance(m, (nn.Conv2d, nn.Linear)):
                            for hook in ddp_forward_hooks:
                                h = m.register_forward_hook(hook)
                                self.ddp_hook_handles.append(h)

                if update_bn:
                    assert enable_quant
                    assert not enable_fp
                    ddp_forward_hooks = set(self.ddp_forward_hooks)
                    for n, m in self.named_modules():
                        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                            for hook in ddp_forward_hooks:
                                h = m.register_forward_hook(hook)
                                self.ddp_hook_handles.append(h)

                if enable_fp:
                    args_fp, kwargs_fp = map_input_to_proper_device(*args, is_quant=False, **kwargs)
                    logits_fp = func(self, *args_fp, **kwargs_fp)
                else:
                    logits_fp = None

                if enable_quant:
                    args_q, kwargs_q = map_input_to_proper_device(*args, is_quant=True, **kwargs)
                    prepare_quant_forward()
                    logits_q = func(self, *args_q, **kwargs_q).to(self.fp_device)
                    recover_fp_forward()
                else:
                    logits_q = None

                if len(self.ddp_hook_handles) > 0:
                    for h in self.ddp_hook_handles:
                        h.remove()
                    self.ddp_hook_handles.clear()

                return logits_fp, logits_q

            return _wrapper

        return _decorate

    @torch.no_grad()
    def update_quant_param(self, calibration_loader, calibration_steps, gamma=0.999, update_bn=False):
        self._update_weight_quant_param()
        self._prepare_calibration_hooks(gamma, update_bn)
        for step, (img, label) in enumerate(calibration_loader):
            if step > calibration_steps:
                break
            _ = self(img, enable_quant=False, update_quant_param=True)
        self.ddp_forward_hooks.clear()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def update_ddp_quant_param(self, model, calibration_loader, calibration_steps, gamma=0.999, update_bn=False):
        # TODO: add `update_bn` to `_prepare_act_quant_param_hook`
        assert isinstance(model, nn.parallel.DistributedDataParallel)
        assert isinstance(model.module, IDQ)
        model_without_ddp = model.module
        model_without_ddp._update_weight_quant_param()
        model_without_ddp._prepare_calibration_hooks(gamma, update_bn)
        logger = getLogger("global")
        for step, (img, label) in enumerate(calibration_loader):
            if step < calibration_steps:
                _ = model(img, enable_quant=False, update_quant_param=True)
                logger.debug(f"[calib step {step:2d}]: max GRAM: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f}MB")
            elif update_bn and step < calibration_steps * 2:
                _ = model(img, enable_fp=False, update_bn=True)
                logger.debug(f"[update BN step {step:2d}]: max GRAM: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f}MB")
            else:
                break
        model_without_ddp.ddp_forward_hooks.clear()
        torch.cuda.empty_cache()

    def get_param_group(self, weight_conf, quant_param_conf, ft_layers=None):
        weight_group = copy(weight_conf)
        quant_param_group = copy(quant_param_conf)
        weight_group["params"] = []
        quant_param_group["params"] = []

        for n, p in self.named_parameters():
            if ft_layers is not None:
                if not isinstance(ft_layers, (list, tuple)):
                    ft_layers = (ft_layers, )

                logger = getLogger("global")
                if any(l in n for l in ft_layers) and "quant_param" not in n:
                    weight_group["params"].append(p)
                    logger.debug(f"finetune: add {n} into optimizer")
                else:
                    logger.debug(f"finetune: skip {n}")
                    p.requires_grad = False
            else:
                if "quant_param" in n:
                    quant_param_group["params"].append(p)
                else:
                    weight_group["params"].append(p)

        return weight_group, quant_param_group

    def to_proper_device(self):
        if self.fp_device == self.q_device:
            self.to(self.fp_device, non_blocking=self.non_blocking)
        else:
            for n, m in self.named_modules():
                if "quant_param" in n:
                    m.to(self.q_device, non_blocking=self.non_blocking)
                else:
                    m.to(self.fp_device, non_blocking=self.non_blocking)
