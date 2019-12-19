# -*- coding: utf-8 -*-

import torch.nn as nn

import quant_pack.core.fuse.functional as fused_f
import quant_pack.core.quant.functional as quant_f

__all__ = ["_registered_fused_forward_functions", "_registered_quant_forward_functions"]

_registered_fused_forward_functions = {
    nn.Conv2d: fused_f.fused_conv_bn_forward,
    nn.BatchNorm2d: fused_f.fused_bn_forward,
    nn.Linear: fused_f.fused_conv_bn_forward,
}

_registered_quant_forward_functions = {
    nn.Conv2d: quant_f.quant_conv2d_forward,
    nn.Linear: quant_f.quant_linear_forward,
}
