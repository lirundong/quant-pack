# -*- coding: utf-8 -*-

from copy import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch.distributions import Multinomial

__all__ = ["LR_CIFAR10"]


def inv_sigmoid(x, lb, ub):
    x = torch.clamp(x, lb, ub)
    return - torch.log(1. / x - 1.)


class TernaryConv2d(nn.Conv2d):
    """Implementation of `LR-Nets`(https://arxiv.org/abs/1710.07739)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, quant=False,
                 p_max=0.95, p_min=0.05):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias)

        self.quant = quant
        self.p_max = p_max
        self.p_min = p_min
        self.register_buffer("w_candidate", torch.tensor([-1., 0., 1.]))
        self.p_a = Parameter(torch.zeros_like(self.weight))
        self.p_b = Parameter(torch.zeros_like(self.weight))
        self.reset_p()

    def reset_p(self):
        w = self.weight.data / self.weight.data.std()
        self.p_a.data = inv_sigmoid(self.p_max - (self.p_max - self.p_min) * w.abs(), self.p_min, self.p_max)
        self.p_b.data = inv_sigmoid(0.5 * (1. + w / (1. - torch.sigmoid(self.p_a.data))), self.p_min, self.p_max)

    def forward(self, input):
        if self.quant:
            p_a = torch.sigmoid(self.p_a)
            p_b = torch.sigmoid(self.p_b)
            p_w_0 = p_a
            p_w_pos = p_b * (1. - p_w_0)
            p_w_neg = (1. - p_b) * (1. - p_w_0)
            p = torch.stack([p_w_neg, p_w_0, p_w_pos], dim=-1)
            if self.training:
                w_mean = (p * self.w_candidate).sum(dim=-1)
                w_var = (p * self.w_candidate.pow(2)).sum(dim=-1) - w_mean.pow(2)
                act_mean = F.conv2d(input, w_mean, self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)
                act_var = F.conv2d(input.pow(2), w_var, self.bias, self.stride,
                                   self.padding, self.dilation, self.groups)
                eps = torch.randn_like(act_mean)
                y = act_mean + eps * act_var.sqrt()
            else:
                m = Multinomial(probs=p)
                indices = m.sample().argmax(dim=-1)
                w = self.w_candidate[indices]
                y = F.conv2d(input, w, self.bias, self.stride,
                             self.padding, self.dilation, self.groups)
        else:
            y = super().forward(input)

        return y


class LR_CIFAR10(nn.Module):

    def __init__(self, num_classes=10):
        """ (2×128C3)−MP2−(2×256C3)−MP2−(2×512C3)−MP2−1024FC−Softmax """
        super(LR_CIFAR10, self).__init__()
        self.num_classes = num_classes

        self.features = nn.Sequential(
            # (2×128C3) − MP2
            TernaryConv2d(3, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            TernaryConv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            # (2×256C3) − MP2
            TernaryConv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            TernaryConv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            # (2×512C3) − MP2
            TernaryConv2d(256, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            TernaryConv2d(512, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(4 * 4 * 512, 1024),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes),
        )

    def forward(self, *input):
        x, = input
        f = self.features(x)
        n = f.size(0)
        f = f.reshape(n, -1)
        logits = self.fc(f)

        return logits

    def opt_param_groups(self, quant=False, **opt_conf):
        decay_group = dict(params=[], **opt_conf)
        no_decay_conf = copy(opt_conf)
        no_decay_conf['weight_decay'] = 0.
        no_decay_group = dict(params=[], **no_decay_conf)

        def param_filter(name, module):
            if isinstance(module, TernaryConv2d) and any(s in name for s in ("weight", "p_a", "p_b")):
                if quant:
                    return any(s in name for s in ("p_a", "p_b"))
                else:
                    return "weight" in name
            else:
                return torch.is_tensor(p)

        for m in self.modules():
            for n, p in m._parameters.items():
                if param_filter(n, m):
                    if isinstance(m, nn.Conv2d) and any(s in n for s in ("weight", "p_a", "p_b")):
                        decay_group["params"].append(p)
                    else:
                        no_decay_group["params"].append(p)

        return [decay_group, no_decay_group]

    def quant(self, enable=True):
        for m in self.modules():
            if isinstance(m, TernaryConv2d):
                m.quant = enable

    def full_precision(self):
        self.quant(False)

    def reset_p(self):
        for m in self.modules():
            if isinstance(m, TernaryConv2d):
                m.reset_p()
