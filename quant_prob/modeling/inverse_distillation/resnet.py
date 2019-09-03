# -*- coding: utf-8 -*-

import torch
from torchvision.models.resnet import ResNet, BasicBlock, Bottleneck, model_urls
from torchvision.models.utils import load_state_dict_from_url

from .idq import IDQ

if torch.cuda.is_available():
    from quant_prob.modeling.quantizers.cuda_param_linear_quantizer import cuda_fake_linear_quant
    quantizer = cuda_fake_linear_quant
else:
    from quant_prob.modeling.quantizers.param_linear_quantizer import fake_linear_quant
    quantizer = fake_linear_quant

__all__ = ["resnet18_idq", "resnet50_idq", "resnet101_idq"]


class ResNetIDQ(ResNet, IDQ):

    def __init__(self, block, layers, num_classes=1000, kw=4, ka=4, fp_layers=None, align_zero=True,
                 devices=None, non_blocking=False):
        ResNet.__init__(self, block, layers, num_classes)
        IDQ.__init__(self, ResNet.forward, kw, ka, fp_layers, align_zero, devices, non_blocking)


def _resnet_idq(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNetIDQ(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18_idq(pretrained=False, progress=True, **kwargs):
    return _resnet_idq("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet50_idq(pretrained=False, progress=True, **kwargs):
    return _resnet_idq("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101_idq(pretrained=False, progress=True, **kwargs):
    return _resnet_idq("resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs)
