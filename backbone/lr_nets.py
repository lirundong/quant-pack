# -*- coding: utf-8 -*-

import torch.nn as nn

from ._components import TernaryConv2d, TernaryLinear, NaiveQuantConv2d, \
                         NaiveQuantLinear, QConv2dDiffBounds, \
                         QLinearDiffBounds, NonLocal
from ._quant_backbone import QuantBackbone

__all__ = ["cifar10_ternary_lr", "cifar10_quant_ste", "cifar10_opt_bounds"]


class CIFAR10(QuantBackbone):
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
            # TODO: all-negative before this, on 2-bits w/o FT
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


def cifar10_ternary_lr(**kwargs):
    return CIFAR10(TernaryConv2d, TernaryLinear, nn.ReLU, **kwargs)


def cifar10_quant_ste(denoise=False, **kwargs):
    if denoise:
        non_linear = NonLocal
    else:
        non_linear = nn.ReLU
    return CIFAR10(NaiveQuantConv2d, NaiveQuantLinear, non_linear, **kwargs)


def cifar10_opt_bounds(**kwargs):
    return CIFAR10(QConv2dDiffBounds, QLinearDiffBounds, nn.ReLU, **kwargs)
