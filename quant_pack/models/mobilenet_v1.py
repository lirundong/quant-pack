# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

__all__ = ["mobilenet_v1"]


def conv_bn(inp, oup, stride, width_mult=1.0):
    inp = int(inp * width_mult)
    oup = int(oup * width_mult)
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True)
    )


def conv_dw(inp, oup, stride, width_mult=1.0):
    inp = int(inp * width_mult)
    oup = int(oup * width_mult)
    return nn.Sequential(
        nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
        nn.BatchNorm2d(inp),
        nn.ReLU(inplace=True),

        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU(inplace=True),
    )


class MobileNetV1(nn.Module):
    """Adapted from: https://github.com/marvis/pytorch-mobilenet"""
    def __init__(self, num_classes=1000, width_mult=1.0):
        super(MobileNetV1, self).__init__()

        self.model = nn.Sequential(
            conv_bn(3, 32, 2, width_mult),
            conv_dw(32, 64, 1, width_mult),
            conv_dw(64, 128, 2, width_mult),
            conv_dw(128, 128, 1, width_mult),
            conv_dw(128, 256, 2, width_mult),
            conv_dw(256, 256, 1, width_mult),
            conv_dw(256, 512, 2, width_mult),
            conv_dw(512, 512, 1, width_mult),
            conv_dw(512, 512, 1, width_mult),
            conv_dw(512, 512, 1, width_mult),
            conv_dw(512, 512, 1, width_mult),
            conv_dw(512, 512, 1, width_mult),
            conv_dw(512, 1024, 2, width_mult),
            conv_dw(1024, 1024, 1, width_mult),
            nn.AvgPool2d(7),
        )
        self.c_fc = int(1024 * width_mult)
        self.fc = nn.Linear(self.c_fc, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.reshape(-1, self.c_fc)
        x = self.fc(x)
        return x


def mobilenet_v1(num_classes=1000, width_mult=1.0, pre_trained=None):
    model = MobileNetV1(num_classes, width_mult)
    if pre_trained:
        ckpt = torch.load(pre_trained, torch.device("cpu"))
        model.load_state_dict(ckpt)
    return model
