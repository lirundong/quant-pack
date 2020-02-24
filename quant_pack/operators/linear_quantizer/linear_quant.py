# -*- coding: utf-8 -*-

import torch
from torch.autograd import Function

from quant_pack.operators.linear_quantizer._C import linear_quant_forward, linear_quant_backward


class LinearQuantFunc(Function):

    @staticmethod
    def forward(ctx, x, lb, ub, bit_width, align_zero):
        with torch.no_grad():
            assert lb.lt(ub).all(), f"invalid quantization range: lb={lb.max().item()}, ub={ub.min().item()}"
        x = x.contiguous()
        channel_quant = lb.dim() > 0
        qx, di, mask_x = linear_quant_forward(x, lb, ub, bit_width, align_zero, channel_quant)
        ctx.save_for_backward(di, mask_x, lb.sign())
        ctx.cfg = (bit_width, align_zero, channel_quant)
        return qx

    @staticmethod
    def backward(ctx, dy):
        dy = dy.clone()
        di, mask_x, sign_lb = ctx.saved_tensors
        bit_width, align_zero, channel_quant = ctx.cfg
        dx, dlb, dub = linear_quant_backward(dy, di, mask_x, sign_lb, bit_width, align_zero, channel_quant)
        return dx, dlb, dub, None, None


class RoundSTE(Function):

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, dx):
        return dx.clone()


class AlignZeroSTE(Function):

    @staticmethod
    def forward(ctx, lb, ub, delta, n):
        if 0 < lb:
            lb = torch.zeros_like(lb)
            ub = lb + delta * n
            zero_point = lb
        elif ub < 0:
            ub = torch.zeros_like(ub)
            lb = ub - delta * n
            zero_point = ub
        else:
            zero_point = lb.abs().div_(delta).round_()
            lb = (-zero_point) * delta
            ub = (n - zero_point) * delta
        ctx.mark_non_differentiable(zero_point)
        return lb, ub, zero_point

    @staticmethod
    def backward(ctx, dlb, dub, d_zero_point):
        return dlb.clone(), dub.clone(), None, None


def autograd_linear_quant(x, lb, ub, bit_width, align_zero):
    with torch.no_grad():
        assert lb.lt(ub).all(), f"invalid quantization range: lb={lb.max().item()}, ub={ub.min().item()}"
        n = 2 ** bit_width - 1
        delta = (ub - lb) / n
    if align_zero:
        lb, ub, zp = AlignZeroSTE.apply(lb, ub, delta, n)
