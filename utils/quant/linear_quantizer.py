# -*- coding: utf-8 -*-

import torch
from torch.autograd import Function

from ._extensions import ext

__all__ = ["fake_linear_quant"]


class LinearQuantSTE(Function):

    @staticmethod
    def forward(ctx, x, lb, ub, bit_width, align_zero):
        x = x.contiguous()
        qx, di, mask_x = ext.linear_quant_forward(x, lb, ub, bit_width, align_zero)
        ctx.save_for_backward(di, mask_x)
        ctx.cfg = (bit_width, lb.sign().item(), align_zero)
        return qx

    @staticmethod
    def backward(ctx, dy):
        dy = dy.clone()
        di, mask_x = ctx.saved_tensors
        bit_width, sign_lb, align_zero = ctx.cfg
        dx, dlb, dub = ext.linear_quant_backward(dy, di, mask_x, bit_width, sign_lb, align_zero)
        return dx, dlb, dub, None, None


def fake_linear_quant(x, lb, ub, k, align_zero):
    assert lb.lt(ub).all(), f"invalid quantization range: lb={lb.max().item()}, ub={ub.min().item()}"
    quantizer = LinearQuantSTE.apply
    return quantizer(x, lb, ub, k, align_zero)
