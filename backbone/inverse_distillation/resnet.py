# -*- coding: utf-8 -*-

import math
from copy import copy
from types import MethodType
from logging import getLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn import Parameter
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

from ._quantizer import fake_quant

__all__ = ["resnet18_idq", "resnet50_idq"]


class ResNetIDQ(ResNet):

    def __init__(self, block, layers, num_classes=1000, kw=4, ka=4, quant_all=True, align_zero=True):
        super(ResNetIDQ, self).__init__(block, layers, num_classes)
        self.kw = kw
        self.ka = ka
        self.quant_all = quant_all
        self.align_zero = align_zero
        self.weight_quant_param = nn.ParameterDict()
        self.activation_quant_param = nn.ParameterDict()
        self.layer_names = dict()
        self.reset_weight_param()
        self.init_quant_param()

    def reset_weight_param(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    fan_in, _ = init._calculate_fan_in_and_fan_out(m.weight)
                    bound = 1 / math.sqrt(fan_in)
                    init.uniform_(m.bias, -bound, bound)

    def init_quant_param(self):
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
    def update_stat(input, name, percentile, param_bank):
        if percentile == 1.:
            v = input.max()
        elif percentile == 0.:
            v = input.min()
        else:
            k = int(input.numel() * percentile)
            v = input[k]
        if name in param_bank:
            criteria = torch.min if percentile < 0.5 else torch.max
            v = criteria(param_bank[name].detach(), v)
        param_bank[name].data = v

    def update_weight_quant_param(self):
        for n, m in self.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                lb = m.weight.detach().min()
                ub = m.weight.detach().max()
                self.weight_quant_param[f"{n}_weight_lb".replace(".", "_")].data = lb
                self.weight_quant_param[f"{n}_weight_ub".replace(".", "_")].data = ub

    def update_activation_quant_param(self, calibration_loader, calibration_steps, gamma=0.999):

        def update_act_stat_hook(module, input, output):
            name = self.layer_names[id(module)]
            if not torch.is_tensor(input):
                assert len(input) == 1
                input = input[0]
            input_view, _ = input.detach().view(-1).sort()
            self.update_stat(input_view, f"{name}_act_ub".replace(".", "_"), gamma, self.activation_quant_param)
            self.update_stat(input_view, f"{name}_act_lb".replace(".", "_"), 1. - gamma, self.activation_quant_param)

        handles = list()
        for n, m in self.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                h = m.register_forward_hook(update_act_stat_hook)
                handles.append(h)

        device = next(self.parameters()).device
        with torch.no_grad():
            for step, (img, label) in enumerate(calibration_loader):
                if step > calibration_steps:
                    break
                _ = self(img.to(device, non_blocking=True), enable_quant=False)

        for h in handles:
            h.remove()

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

    def forward(self, inputs, enable_quant=True, update_stat=False):

        def do_fake_quant(m, x, detach_w=False, gamma=0.999):  # TODO: add this to interface
            name = self.layer_names[id(m)]
            if update_stat:
                input_view, _ = x.detach().view(-1).sort()
                weight_view = m.weight.detach().view(-1)
                self.update_stat(input_view, f"{name}_act_ub".replace(".", "_"), gamma, self.activation_quant_param)
                self.update_stat(input_view, f"{name}_act_lb".replace(".", "_"), 1. - gamma, self.activation_quant_param)
                self.update_stat(weight_view, f"{name}_weight_ub".replace(".", "_"), 1., self.weight_quant_param)
                self.update_stat(weight_view, f"{name}_weight_lb".replace(".", "_"), 0., self.weight_quant_param)

            w_lb = self.weight_quant_param[f"{name}_weight_lb".replace(".", "_")]
            w_ub = self.weight_quant_param[f"{name}_weight_ub".replace(".", "_")]
            x_lb = self.activation_quant_param[f"{name}_act_lb".replace(".", "_")]
            x_ub = self.activation_quant_param[f"{name}_act_ub".replace(".", "_")]
            w = m.weight.detach() if detach_w else m.weight
            qw = fake_quant(w, w_lb, w_ub, self.kw, self.align_zero)
            qx = fake_quant(x, x_lb, x_ub, self.ka, self.align_zero)
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
                if not self.quant_all and (n == "conv1" or n == "fc"):
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

        logits_fp = super(ResNetIDQ, self).forward(inputs)
        if enable_quant:
            prepare_quant_forward()
            logits_q = super(ResNetIDQ, self).forward(inputs)
            recover_fp_forward()
        else:
            logits_q = None

        return logits_fp, logits_q


def resnet18_idq(**kwargs):
    return ResNetIDQ(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet50_idq(**kwargs):
    return ResNetIDQ(Bottleneck, [3, 4, 6, 3], **kwargs)
