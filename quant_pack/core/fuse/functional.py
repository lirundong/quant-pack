# -*- coding: utf-8 -*-

from distutils.version import LooseVersion

import torch
import torch.nn.functional as F

__all__ = ["fused_conv_bn_forward", "fused_bn_forward", "fused_fc_bn_forward"]


def _var_mean(input, dim, unbiased=True):
    var = torch.var(input, dim=dim, unbiased=unbiased)
    mean = torch.mean(input, dim=dim)
    return var, mean


if LooseVersion(torch.__version__) >= LooseVersion("1.2"):
    var_mean = torch.var_mean
else:
    var_mean = _var_mean


def fused_conv_bn_forward(module, input):
    # TODO: assertion of `input_transform` and `weight_transform`
    if module.input_transform is not None:
        input = module.input_transform(input)

    with torch.no_grad():
        if module.training:
            pre_activation = F.conv2d(input, module.weight, module.bias, module.stride,
                                      module.padding, module.dilation, module.groups)
            pre_activation = pre_activation.permute(1, 0, 2, 3).reshape(module.out_channels, -1)
            var, mean = var_mean(pre_activation, dim=1)
            module.running_mean.mul_(1. - module.bn_momentum).add_(module.bn_momentum, mean)
            module.running_var.mul_(1. - module.bn_momentum).add_(module.bn_momentum, var)
        else:
            var, mean = module.running_var, module.running_mean
        safe_std = torch.sqrt(var + module.bn_eps)
        w_view = (module.out_channels, 1, 1, 1)

    if module.affine:
        weight = module.weight * (module.alpha / safe_std).view(w_view)
        beta = module.beta - module.alpha * mean / safe_std
        if module.bias is not None:
            bias = module.alpha * module.bias / safe_std + beta
        else:
            bias = beta
    else:
        weight = module.weight / safe_std.view(w_view)
        beta = -mean / safe_std
        if module.bias is not None:
            bias = module.bias / safe_std + beta
        else:
            bias = beta

    if module.weight_transform is not None:
        weight = module.weight_transform(weight)

    return F.conv2d(input, weight, bias, module.stride,
                    module.padding, module.dilation, module.groups)


def fused_bn_forward(module, input):
    return input


def fused_fc_bn_forward(module, input):
    raise NotImplementedError()
