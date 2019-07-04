# -*- coding: utf-8 -*-

import math
from copy import copy
from types import MethodType
from logging import getLogger

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

if torch.cuda.is_available():
    from utils.quant.linear_quantizer import fake_linear_quant
    _quantizer = fake_linear_quant
else:
    from ._quantizer import fake_quant
    _quantizer = fake_quant

__all__ = ["resnet18_idq", "resnet50_idq"]


class ResNetIDQ(ResNet):

    # TODO: wrap IDQ into a class decorator

    def __init__(self, block, layers, num_classes=1000, kw=4, ka=4, quant_all=True, align_zero=True):
        super(ResNetIDQ, self).__init__(block, layers, num_classes)
        self.kw = kw
        self.ka = ka
        self.quant_all = quant_all
        self.align_zero = align_zero
        self.weight_quant_param = nn.ParameterDict()
        self.activation_quant_param = nn.ParameterDict()
        self.layer_names = dict()

        # Since normal hooks do not invoked if module is wrapped by DDP, we
        # re-initialize them in `forward` at specific iterations. Currently we
        # only used `forward_hooks`.
        self.ddp_forward_hooks = []
        self.ddp_hook_handles = []

        self._reset_weight_param()
        self._init_quant_param()

    def _reset_weight_param(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    init.uniform_(m.bias, -bound, bound)
            elif isinstance(m, nn.BatchNorm2d):
                if m.track_running_stats:
                    m.running_mean.zero_()
                    m.running_var.fill_(1)
                    m.num_batches_tracked.zero_()
                if m.affine:
                    init.uniform_(m.weight)
                    init.zeros_(m.bias)

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

    @staticmethod
    def _update_stat(input, name, percentile, param_bank):
        assert torch.is_tensor(input)
        if input.requires_grad:
            input = input.detach()
        if percentile == 1.:
            v = input.max()
        elif percentile == 0.:
            v = input.min()
        else:
            assert input.dim() == 1
            k = int(math.floor(input.numel() * percentile))
            v = input[k]
        if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
            dist.barrier()
            dist.all_reduce(v)
            v.div_(dist.get_world_size())
        try:
            param_bank[name].data = v
        except KeyError as e:
            raise RuntimeError(f"update quant-param which not seen in init: `{e}`")

    def _update_weight_quant_param(self):
        for n, m in self.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                weight_view = m.weight.detach().view(-1)
                self._update_stat(weight_view, f"{n}_weight_lb".replace(".", "_"), 0., self.weight_quant_param)
                self._update_stat(weight_view, f"{n}_weight_ub".replace(".", "_"), 1., self.weight_quant_param)

    def _prepare_act_quant_param_hook(self, gamma=0.999):

        def update_act_stat_hook(module, input, output):
            name = self.layer_names[id(module)]
            if not torch.is_tensor(input):
                assert len(input) == 1
                input = input[0]
            input_view, _ = input.detach().view(-1).sort()
            self._update_stat(input_view, f"{name}_act_ub".replace(".", "_"), gamma, self.activation_quant_param)
            self._update_stat(input_view, f"{name}_act_lb".replace(".", "_"), 1. - gamma, self.activation_quant_param)

        assert len(self.ddp_forward_hooks) == 0
        self.ddp_forward_hooks.append(update_act_stat_hook)

    @torch.no_grad()
    def update_quant_param(self, calibration_loader, calibration_steps, gamma=0.999):
        self._update_weight_quant_param()
        self._prepare_act_quant_param_hook(gamma)
        device = next(self.parameters()).device
        for step, (img, label) in enumerate(calibration_loader):
            if step > calibration_steps:
                break
            _ = self(img.to(device, non_blocking=True), enable_quant=False, update_quant_param=True)
        self.ddp_forward_hooks.clear()

    @staticmethod
    @torch.no_grad()
    def update_ddp_quant_param(model, calibration_loader, calibration_steps, gamma=0.999):
        assert isinstance(model, nn.parallel.DistributedDataParallel)
        assert isinstance(model.module, ResNetIDQ)
        model_without_ddp = model.module
        model_without_ddp._update_weight_quant_param()
        model_without_ddp._prepare_act_quant_param_hook(gamma)
        device = next(model.parameters()).device
        logger = getLogger("global")
        for step, (img, label) in enumerate(calibration_loader):
            if step > calibration_steps:
                break
            _ = model(img.to(device, non_blocking=True), enable_quant=False, update_quant_param=True)
            logger.debug(f"[calib step {step:2d}]: max GRAM: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f}MB")
        model_without_ddp.ddp_forward_hooks.clear()

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

    def forward(self, inputs, enable_quant=True, update_quant_param=False):

        def do_fake_quant(m, x):  # TODO: add this to interface
            name = self.layer_names[id(m)]
            w_lb = self.weight_quant_param[f"{name}_weight_lb".replace(".", "_")]
            w_ub = self.weight_quant_param[f"{name}_weight_ub".replace(".", "_")]
            x_lb = self.activation_quant_param[f"{name}_act_lb".replace(".", "_")]
            x_ub = self.activation_quant_param[f"{name}_act_ub".replace(".", "_")]
            qw = _quantizer(m.weight, w_lb, w_ub, self.kw, self.align_zero)
            qx = _quantizer(x, x_lb, x_ub, self.ka, self.align_zero)
            return qx, qw

        def quant_conv2d_forward(m, x):
            assert isinstance(m, nn.Conv2d)
            qx, qw = do_fake_quant(m, x)
            return F.conv2d(qx, qw, m.bias, m.stride, m.padding, m.dilation, m.groups)

        def quant_linear_forward(m, x):
            assert isinstance(m, nn.Linear)
            qx, qw = do_fake_quant(m, x)
            return F.linear(qx, qw, m.bias)

        def prepare_quant_forward():
            for n, m in self.named_modules():
                if not self.quant_all and ("conv1" in n or "fc" in n):
                    continue
                if isinstance(m, nn.Conv2d):
                    m.forward = MethodType(quant_conv2d_forward, m)
                elif isinstance(m, nn.Linear):
                    m.forward = MethodType(quant_linear_forward, m)

        def recover_fp_forward():
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    m.forward = MethodType(nn.Conv2d.forward, m)
                elif isinstance(m, nn.Linear):
                    m.forward = MethodType(nn.Linear.forward, m)

        if update_quant_param:
            assert len(self.ddp_forward_hooks) > 0 and len(self.ddp_hook_handles) == 0
            ddp_forward_hooks = set(self.ddp_forward_hooks)
            for n, m in self.named_modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    for hook in ddp_forward_hooks:
                        h = m.register_forward_hook(hook)
                        self.ddp_hook_handles.append(h)

        logits_fp = super(ResNetIDQ, self).forward(inputs)
        if enable_quant:
            prepare_quant_forward()
            logits_q = super(ResNetIDQ, self).forward(inputs)
            recover_fp_forward()
        else:
            logits_q = None

        if update_quant_param:
            for h in self.ddp_hook_handles:
                h.remove()
            self.ddp_hook_handles.clear()

        return logits_fp, logits_q


def resnet18_idq(**kwargs):
    return ResNetIDQ(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet50_idq(**kwargs):
    return ResNetIDQ(Bottleneck, [3, 4, 6, 3], **kwargs)
