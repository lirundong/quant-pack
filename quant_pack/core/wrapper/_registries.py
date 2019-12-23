# -*- coding: utf-8 -*-

import torch.nn as nn

import quant_pack.core.fuse.functional as fused_f
import quant_pack.core.quant.functional as quant_f

__all__ = ["FUSED_FORWARD_FUNCTIONS", "QUANT_FORWARD_FUNCTIONS"]

FUSED_FORWARD_FUNCTIONS = {
    nn.Conv2d: fused_f.fused_conv_bn_forward,
    nn.BatchNorm2d: fused_f.fused_bn_forward,
    nn.Linear: fused_f.fused_conv_bn_forward,
}

QUANT_FORWARD_FUNCTIONS = {
    nn.Conv2d: quant_f.quant_conv2d_forward,
    nn.Linear: quant_f.quant_linear_forward,
}
