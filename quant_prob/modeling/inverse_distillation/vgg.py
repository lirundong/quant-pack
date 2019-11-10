# -*- coding: utf-8 -*-

import torch
import torch.nn as nn

from .idq import IDQ

__all__ = ["cifar10_vgg7", "cifar10_vgg7_idq"]

_vgg_conf = {
    # CIFAR-10 input size: 32
    # 2x(128C3) - MP2 - 2x(256C3) - MP2 - 2x(512C3) - MP2 - 1024FC - Softmax
    "vgg7_cifar10": [128, 128, "M", 256, 256, "M", 512, 512],
}


def _make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class CIFAR_VGG(nn.Module):

    def __init__(self, features, num_classes=10, init_weights=True):
        super(CIFAR_VGG, self).__init__()
        self.features = features
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.classifier = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(1024, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


class CIFAR_VGG_IDQ(CIFAR_VGG, IDQ):

    def __init__(self, features, num_classes=10, init_weights=True,
                 kw=4, ka=4, fp_layers=None, align_zero=True,
                 use_channel_quant=False, use_ckpt=False, use_multi_domain=False):
        CIFAR_VGG.__init__(self, features, num_classes, init_weights)
        IDQ.__init__(self, CIFAR_VGG.forward, kw, ka, fp_layers,
                     align_zero, use_channel_quant, use_ckpt, use_multi_domain)


def cifar10_vgg7():
    return CIFAR_VGG(_make_layers(_vgg_conf["vgg7_cifar10"], batch_norm=True))


def cifar10_vgg7_idq(**kwargs):
    return CIFAR_VGG_IDQ(_make_layers(_vgg_conf["vgg7_cifar10"], batch_norm=True), **kwargs)
