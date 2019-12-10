# -*- coding: utf-8 -*-

import torch
from torch.autograd import Function

from quant_pack.extensions import ext

__all__ = ["cuda_fake_linear_quant"]


class LinearQuantSTE(Function):

    @staticmethod
    def forward(ctx, x, lb, ub, bit_width, align_zero):
        x = x.contiguous()
        channel_quant = lb.dim() > 0
        qx, di, mask_x = ext.linear_quant_forward(x, lb, ub, bit_width, align_zero, channel_quant)
        ctx.save_for_backward(di, mask_x, lb.sign())
        ctx.cfg = (bit_width, align_zero, channel_quant)
        return qx

    @staticmethod
    def backward(ctx, dy):
        dy = dy.clone()
        di, mask_x, sign_lb = ctx.saved_tensors
        bit_width, align_zero, channel_quant = ctx.cfg
        dx, dlb, dub = ext.linear_quant_backward(dy, di, mask_x, sign_lb, bit_width, align_zero, channel_quant)
        return dx, dlb, dub, None, None


def cuda_fake_linear_quant(x, lb, ub, k, align_zero):
    if k == 32:
        return x
    else:
        with torch.no_grad():
            assert lb.lt(ub).all(), f"invalid quantization range: lb={lb.max().item()}, ub={ub.min().item()}"
        quantizer = LinearQuantSTE.apply
        return quantizer(x, lb, ub, k, align_zero)
