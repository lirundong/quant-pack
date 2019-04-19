# -*- coding: utf-8 -*-

from copy import copy
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck

from ._quantizer import fake_quant

__all__ = ["resnet18_idq", "resnet50_idq"]


class ResNetIDQ(ResNet):

    def __init__(self, block, layers, num_classes=1000, bit_width=4):
        super(ResNetIDQ, self).__init__(block, layers, num_classes)
        self.bit_width = bit_width
        # TODO: examine the grad_hook on these parameters
        self.weight_quant_param = nn.ParameterDict()
        self.activation_quant_param = nn.ParameterDict()
        self.layer_names = dict()
        self.init_quant_param()

    def init_quant_param(self):
        for n, m in self.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                lb = Parameter(m.weight.detach().min())
                ub = Parameter(m.weight.detach().max())
                self.weight_quant_param[f"{n}_weight_lb".replace(".", "_")] = lb
                self.weight_quant_param[f"{n}_weight_ub".replace(".", "_")] = ub
                self.activation_quant_param[f"{n}_act_lb".replace(".", "_")] = Parameter(torch.tensor(0.))
                self.activation_quant_param[f"{n}_act_ub".replace(".", "_")] = Parameter(torch.tensor(0.))

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

            def _update_stat(n, p):
                k = int(input_view.numel() * p)
                # v, _ = input_view.kthelement(k)
                v = input_view[k]
                if n in self.activation_quant_param:
                    criteria = torch.min if p < 0.5 else torch.max
                    v = criteria(self.activation_quant_param[n].detach(), v)
                self.activation_quant_param[n].data = v

            _update_stat(f"{name}_act_ub".replace(".", "_"), gamma)
            _update_stat(f"{name}_act_lb".replace(".", "_"), 1. - gamma)

        handles = list()
        for n, m in self.named_modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                self.layer_names[id(m)] = n
                h = m.register_forward_hook(update_act_stat_hook)
                handles.append(h)

        device = next(self.parameters()).device
        with torch.no_grad():
            for step, (img, label) in enumerate(calibration_loader):
                if step > calibration_steps:
                    break
                _ = self(img.to(device, non_blocking=True), fp_only=True)

        for h in handles:
            h.remove()

    def get_param_group(self, **opt_conf):
        decay_group = copy(opt_conf)
        decay_group["params"] = []
        no_decay_group = copy(opt_conf)
        no_decay_group["weight_decay"] = 0.
        no_decay_group["params"] = []

        for m in self.modules():
            if len(m._parameters) > 0:  # TODO: dirty hack
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    decay_group["params"] += list(m.parameters())
                else:
                    no_decay_group["params"] += list(m.parameters())

        return decay_group, no_decay_group

    def forward(self, inputs, fp_only=False):

        def do_fake_quant(m, x):
            n = self.layer_names[id(m)]
            w_lb = self.weight_quant_param[f"{n}_weight_lb".replace(".", "_")]
            w_ub = self.weight_quant_param[f"{n}_weight_ub".replace(".", "_")]
            x_lb = self.activation_quant_param[f"{n}_act_lb".replace(".", "_")]
            x_ub = self.activation_quant_param[f"{n}_act_ub".replace(".", "_")]
            qw = fake_quant(m.weight, w_lb, w_ub, self.bit_width)
            qx = fake_quant(x, x_lb, x_ub, self.bit_width)
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
            for m in self.modules():
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
        if fp_only:
            logits_q = None
        else:
            prepare_quant_forward()
            logits_q = super(ResNetIDQ, self).forward(inputs)
            recover_fp_forward()

        return logits_fp, logits_q


def resnet18_idq(**kwargs):
    return ResNetIDQ(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet50_idq(**kwargs):
    return ResNetIDQ(Bottleneck, [3, 4, 6, 3], **kwargs)
