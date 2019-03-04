# -*- coding: utf-8 -*-

from copy import copy

import torch
import torch.nn as nn

from ._components import TernaryConv2d, TernaryLinear, NaiveQuantConv2d, NaiveQuantLinear, \
    NonLocal

__all__ = ["cifar10_ternary_lr", "cifar10_quant_ste"]

quantifiable = (TernaryConv2d, TernaryLinear, NaiveQuantConv2d, NaiveQuantLinear)
visible = (TernaryConv2d, TernaryLinear, NaiveQuantConv2d, NaiveQuantLinear)
probabilistic = (TernaryConv2d, TernaryLinear)
decayable = (nn.Conv2d, nn.Linear)
param_denoise = (NonLocal, )


class CIFAR10(nn.Module):

    # convolution blocks: num_block, num_channel
    cfg = [
        (2, 128),
        (2, 256),
        (2, 512),
    ]

    def __init__(self, conv_block, fc_block, non_linear, num_classes=10, **kwargs):
        """ (2×128C3)−MP2−(2×256C3)−MP2−(2×512C3)−MP2−1024FC−Softmax """
        super(CIFAR10, self).__init__()
        self.conv_block = conv_block
        self.fc_block = fc_block
        self.non_linear = non_linear
        self.num_classes = num_classes

        conv_blocks = []
        prev_c = 3
        for n, c in self.cfg:
            for i in range(n):
                conv_blocks += [
                    self.conv_block(prev_c, c, kernel_size=3, padding=1, bias=False, **kwargs),
                    nn.BatchNorm2d(c),
                ]
                if self.non_linear is nn.ReLU:
                    conv_blocks.append(self.non_linear(inplace=True))
                elif self.non_linear is NonLocal:
                    conv_blocks.append(self.non_linear(c, inplace=True))
                prev_c = c
            conv_blocks.append(nn.MaxPool2d(2, stride=2))

        self.features = nn.Sequential(*conv_blocks)
        self.fc = nn.Sequential(
            nn.Dropout(p=0.5),
            self.fc_block(4 * 4 * 512, 1024, **kwargs),
            nn.ReLU(inplace=True),
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

    def opt_param_groups(self, opt_prob=False, denoise_only=False, **opt_conf):
        decay_group = dict(params=[], **opt_conf)
        no_decay_conf = copy(opt_conf)
        no_decay_conf['weight_decay'] = 0.
        no_decay_group = dict(params=[], **no_decay_conf)

        if denoise_only:
            for m in self.modules():
                if isinstance(m, param_denoise):
                    for p in m.parameters():
                        no_decay_group["params"].append(p)
            return [no_decay_group, ]

        def param_filter(name, module):
            if isinstance(module, probabilistic) and any(s in name for s in ("weight", "p_a", "p_b")):
                if opt_prob:
                    return any(s in name for s in ("p_a", "p_b"))
                else:
                    return "weight" in name
            else:
                return torch.is_tensor(p)

        for m in self.modules():
            for n, p in m._parameters.items():
                if param_filter(n, m):
                    if isinstance(m, decayable) and any(s in n for s in ("weight", "p_a", "p_b")):
                        decay_group["params"].append(p)
                    else:
                        no_decay_group["params"].append(p)

        return [decay_group, no_decay_group]

    def quant(self, enable=True):
        for m in self.modules():
            if isinstance(m, quantifiable):
                m.quant = enable

    def full_precision(self):
        self.quant(False)

    def reset_p(self):
        for m in self.modules():
            if isinstance(m, probabilistic):
                m.reset_p()

    def register_vis(self, tb_logger):
        for n, m in self.named_modules():
            if isinstance(m, visible):
                m.name = n
                m.tb_logger = tb_logger

    def vis(self, iter):
        for n, m in self.named_modules():
            if isinstance(m, visible):
                m.iter = iter
                m.vis = True


def cifar10_ternary_lr(**kwargs):
    return CIFAR10(TernaryConv2d, TernaryLinear, **kwargs)


def cifar10_quant_ste(denoise=False, **kwargs):
    if denoise:
        non_linear = NonLocal
    else:
        non_linear = nn.ReLU
    return CIFAR10(NaiveQuantConv2d, NaiveQuantLinear, non_linear, **kwargs)
