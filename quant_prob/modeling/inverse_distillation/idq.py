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
from torch.utils.checkpoint import checkpoint

if torch.cuda.is_available():
    from quant_prob.modeling.quantizers.cuda_param_linear_quantizer import cuda_fake_linear_quant as quantizer
else:
    from quant_prob.modeling.quantizers.param_linear_quantizer import fake_linear_quant as quantizer

__all__ = ["IDQ"]


class IDQ:

    def __init__(self, forward_func, kw=4, ka=4, fp_layers=None, align_zero=True, use_channel_quant=False,
                 use_ckpt=False, use_multi_domain=False):
        assert isinstance(self, nn.Module), f"IDQ should be used in conjunction with `nn.Module`"

        if fp_layers is not None and not isinstance(fp_layers, (list, tuple)):
            fp_layers = (fp_layers,)

        self.kw = kw
        self.ka = ka
        self.fp_layers = fp_layers
        self.align_zero = align_zero
        self.weight_quant_param = nn.ParameterDict()
        self.activation_quant_param = nn.ParameterDict()
        self.layer_names = dict()
        self.use_channel_quant = use_channel_quant
        self.use_ckpt = use_ckpt
        self.use_multi_domain = use_multi_domain

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
        self.in_quant_mode = False

        self._init_quant_param()
        self.reinit_multi_domain()

    def _init_quant_param(self):

        def _param_range(p):
            p = p.detach()
            if self.use_channel_quant:
                c_out = p.size(0)
                p_view = p.reshape(c_out, -1)
                p_min, _ = p_view.min(dim=1)
                p_max, _ = p_view.max(dim=1)
            else:
                p_min = p.min()
                p_max = p.max()
            return Parameter(p_min), Parameter(p_max)

        for n, m in self.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                self.layer_names[id(m)] = n
                lb, ub = _param_range(m.weight)
                self.weight_quant_param[f"{n}_weight_lb".replace(".", "_")] = lb
                self.weight_quant_param[f"{n}_weight_ub".replace(".", "_")] = ub
                self.activation_quant_param[f"{n}_act_lb".replace(".", "_")] = Parameter(torch.tensor(0.))
                self.activation_quant_param[f"{n}_act_ub".replace(".", "_")] = Parameter(torch.tensor(0.))

    def reinit_multi_domain(self):
        if not self.use_multi_domain:
            return

        def _get_running_stat(module):
            if self.use_multi_domain and self.in_quant_mode:
                running_mean = module.running_mean_q
                running_var = module.running_var_q
                num_batches_tracked = module.num_batches_tracked_q
            else:
                running_mean = module.running_mean
                running_var = module.running_var
                num_batches_tracked = module.num_batches_tracked
            return running_mean, running_var, num_batches_tracked

        def _check_input_dim(x):
            if x.dim() <= 2:
                raise ValueError(f'expected at least 3D input (got {x.dim()}D input)')

        def _multi_domain_bn_forward(module, x):
            _check_input_dim(x)
            running_mean, running_var, num_batches_tracked = module.get_running_stat()

            if module.momentum is None:
                exponential_average_factor = 0.0
            else:
                exponential_average_factor = module.momentum

            if module.training and module.track_running_stats:
                num_batches_tracked.add_(1)
                if module.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / num_batches_tracked.item()
                else:  # use exponential moving average
                    exponential_average_factor = module.momentum

            return F.batch_norm(x, running_mean, running_var, module.weight, module.bias,
                                module.training or not module.track_running_stats,
                                exponential_average_factor, module.eps)

        def _multi_domain_sync_bn_forward(module, x):
            if not x.is_cuda:
                raise ValueError('expected x tensor to be on GPU')

            if not module.ddp_gpu_size:
                raise AttributeError('SyncBatchNorm is only supported within torch.nn.parallel.DistributedDataParallel')

            _check_input_dim(x)

            exponential_average_factor = 0.0
            running_mean, running_var, num_batches_tracked = module.get_running_stat()

            if module.training and module.track_running_stats:
                num_batches_tracked.add_(1)
                if module.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / num_batches_tracked.item()
                else:  # use exponential moving average
                    exponential_average_factor = module.momentum

            world_size = 1
            process_group = torch.distributed.group.WORLD
            if module.process_group:
                process_group = module.process_group
            world_size = torch.distributed.get_world_size(process_group)

            # fallback to framework BN when synchronization is not necessary
            if world_size == 1 or (not module.training and module.track_running_stats):
                return F.batch_norm(x, running_mean, running_var, module.weight, module.bias,
                                    module.training or not module.track_running_stats,
                                    exponential_average_factor, module.eps)
            else:
                return sync_batch_norm.apply(
                    x, module.weight, module.bias, running_mean, running_var,
                    module.eps, exponential_average_factor, process_group, world_size)

        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                assert m._version == 2, f"deprecated batchnorm implementation, " \
                                        f"please switch to pytorch>=1.1"
                device = next(m.buffers()).device
                if not hasattr(m, "running_mean_q"):
                    m.register_buffer("running_mean_q", torch.zeros(m.num_features, device=device))
                if not hasattr(m, "running_var_q"):
                    m.register_buffer("running_var_q", torch.ones(m.num_features, device=device))
                if not hasattr(m, "num_batches_tracked_q"):
                    m.register_buffer("num_batches_tracked_q", torch.tensor(0, dtype=torch.long, device=device))

                m.get_running_stat = MethodType(_get_running_stat, m)
                if isinstance(m, nn.BatchNorm2d):
                    m.forward = MethodType(_multi_domain_bn_forward, m)
                elif isinstance(m, nn.SyncBatchNorm):
                    m.forward = MethodType(_multi_domain_sync_bn_forward, m)

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
            param_bank[name].detach_().copy_(v)
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

            running_mean, running_var, _ = module.get_running_stat()
            running_mean.copy_(mean)
            running_var.copy_(var)

        assert len(self.ddp_forward_hooks) == 0
        self.ddp_forward_hooks.append(update_act_stat_hook)
        if update_bn:
            self.ddp_forward_hooks.append(update_bn_stat_hook)

    def _wrap_forward(self):

        def do_fake_quant(name, weight, x):
            w_lb = self.weight_quant_param[f"{name}_weight_lb".replace(".", "_")]
            w_ub = self.weight_quant_param[f"{name}_weight_ub".replace(".", "_")]
            x_lb = self.activation_quant_param[f"{name}_act_lb".replace(".", "_")]
            x_ub = self.activation_quant_param[f"{name}_act_ub".replace(".", "_")]
            qw = quantizer(weight, w_lb, w_ub, self.kw, self.align_zero)
            qx = quantizer(x, x_lb, x_ub, self.ka, self.align_zero)
            return qx, qw

        def quant_conv2d_forward(m, x):
            assert isinstance(m, nn.Conv2d)
            name = self.layer_names[id(m)]
            qx, qw = do_fake_quant(name, m.weight, x)
            return F.conv2d(qx, qw, m.bias, m.stride, m.padding, m.dilation, m.groups)

        def quant_linear_forward(m, x):
            assert isinstance(m, nn.Linear)
            name = self.layer_names[id(m)]
            qx, qw = do_fake_quant(name, m.weight, x)
            return F.linear(qx, qw, m.bias)

        @contextmanager
        def quant_forward():
            self.in_quant_mode = True
            for n, m in self.named_modules():
                if self.fp_layers is not None and any(fp_n in n for fp_n in self.fp_layers):
                    continue
                if isinstance(m, nn.Conv2d):
                    m.forward = MethodType(quant_conv2d_forward, m)
                elif isinstance(m, nn.Linear):
                    m.forward = MethodType(quant_linear_forward, m)
            try:
                yield
            finally:
                self.in_quant_mode = False
                for m in self.modules():
                    if isinstance(m, nn.Conv2d):
                        m.forward = MethodType(nn.Conv2d.forward, m)
                    elif isinstance(m, nn.Linear):
                        m.forward = MethodType(nn.Linear.forward, m)

        def _decorate(func):
            @wraps(func)
            def _wrapper(*args,
                         enable_fp=True,
                         enable_quant=True,
                         update_quant_param=False,
                         update_bn=False,
                         **kwargs):
                if self.use_ckpt:
                    assert len(kwargs) == 0, "torch.checkpoint does not support kwargs"

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
                    if self.use_ckpt and self.training and not update_quant_param:
                        logits_fp = checkpoint(lambda *x: func(self, *x), *args)
                    else:
                        logits_fp = func(self, *args, **kwargs)
                else:
                    logits_fp = None

                if enable_quant:
                    with quant_forward():
                        if self.use_ckpt and self.training and not update_bn:
                            logits_q = checkpoint(lambda *x: func(self, *x), *args)
                        else:
                            logits_q = func(self, *args, **kwargs)
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
    def update_quant_param(self, model, calibration_loader, calibration_steps, gamma=0.999, update_bn=False):
        self._update_weight_quant_param()
        self._prepare_calibration_hooks(gamma, update_bn)
        device = next(iter(self.parameters())).device
        for step, (img, label) in enumerate(calibration_loader):
            if step > calibration_steps:
                break
            _ = model(img.to(device, non_blocking=True), enable_quant=False, update_quant_param=True)
        self.ddp_forward_hooks.clear()

    @torch.no_grad()
    def update_ddp_quant_param(self, model, calibration_loader, calibration_steps, gamma=0.999, update_bn=False):
        # TODO: add `update_bn` to `_prepare_act_quant_param_hook`
        assert isinstance(model, nn.parallel.DistributedDataParallel)
        assert isinstance(model.module, IDQ)
        model_without_ddp = model.module
        model_without_ddp._update_weight_quant_param()
        model_without_ddp._prepare_calibration_hooks(gamma, update_bn)
        device = next(iter(self.parameters())).device
        logger = getLogger("global")
        for step, (img, label) in enumerate(calibration_loader):
            if step < calibration_steps:
                _ = model(img.to(device, non_blocking=True), enable_quant=False, update_quant_param=True)
                logger.debug(
                    f"[calib step {step:2d}]: max GRAM: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f}MB")
            elif update_bn and step < calibration_steps * 2:
                _ = model(img.to(device, non_blocking=True), enable_fp=False, update_bn=True)
                logger.debug(
                    f"[update BN step {step:2d}]: max GRAM: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f}MB")
            else:
                break
        model_without_ddp.ddp_forward_hooks.clear()

    @torch.no_grad()
    def get_activations(self, loader, *names):
        assert not isinstance(self, nn.parallel.DistributedDataParallel)

        act_bank = {}
        handles = []

        def save_activation_hook(m, x, y):
            assert isinstance(m, (nn.Conv2d, nn.Linear))
            subfix = "_q" if self.in_quant_mode else "_fp"
            m_name = self.layer_names[id(m)] + subfix
            y_data = y.detach().cpu().numpy()
            act_bank[m_name] = y_data

        for n, m in self.named_modules():
            if any(k in n for k in names) and isinstance(m, (nn.Conv2d, nn.Linear)):
                h = m.register_forward_hook(save_activation_hook)
                handles.append(h)

        img, label = next(iter(loader))
        _ = self(img.cuda(), enable_quant=True)

        for h in handles:
            h.remove()

        return act_bank

    def get_param_group(self, weight_conf, quant_param_conf, ft_layers=None):
        weight_group = copy(weight_conf)
        quant_param_group = copy(quant_param_conf)
        weight_group["params"] = []
        quant_param_group["params"] = []

        for n, p in self.named_parameters():
            if ft_layers is not None:
                if not isinstance(ft_layers, (list, tuple)):
                    ft_layers = (ft_layers,)

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
