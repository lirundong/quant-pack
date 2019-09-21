# -*- coding: utf-8 -*-

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function

__all__ = ["fake_linear_quant", "clamp"]

QuantT = Tuple[Tensor, Tensor, Tensor]  # I, z, delta


class RoundSTE(Function):

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        return x.round()

    @staticmethod
    def backward(ctx, dy: Tensor) -> Tensor:
        return dy.clone()


class BinaryQuant(Function):

    @staticmethod
    def forward(ctx, x, lb, ub):
        mask_up = x > 0
        bx = torch.where(mask_up, ub, lb)
        ctx.save_for_backward(mask_up)

        return bx

    @staticmethod
    def backward(ctx, dy):
        mask_up, = ctx.saved_tensors
        dy = dy.clone()
        d_lb = dy[~mask_up].sum()
        d_ub = dy[mask_up].sum()

        return dy, d_lb, d_ub


def clamp(x: Tensor, lb: Tensor, ub: Tensor) -> Tensor:

    def _make_broadcast(t):
        if t.dim() == 0 or t.dim() == x.dim():
            return t
        else:
            assert t.dim() == 1
            c = t.size(0)
            dim = x.dim() - 1
            return t.reshape((c, ) + (1, ) * dim)

    lb = _make_broadcast(lb)
    ub = _make_broadcast(ub)
    x = torch.min(x, ub)
    x = torch.max(lb, x)
    return x


def linear_quant(x: Tensor, lb: Tensor, ub: Tensor, k: int) -> QuantT:
    round_ = RoundSTE.apply
    n = 2 ** k - 1
    eps = 1e-2  # TODO: add this to interface?
    ub = torch.max(lb.add(eps), ub)
    delta = ub.sub(lb).div(n)
    z = round_(lb.abs().div(delta))
    lb = z.neg().mul(delta)
    ub = (n - z).mul(delta)
    x = clamp(x, lb, ub)
    i = round_(x.sub(lb).div(delta))

    return i, z, delta


def dequantizer(qx: QuantT) -> Tensor:
    i, z, delta = qx
    return i.sub(z).mul(delta)


def fake_linear_quant(x: Tensor, lb: Tensor, ub: Tensor, k: int, align_zero: bool = True) -> Tensor:
    with torch.no_grad():
        assert lb.lt(ub).all(), f"invalid quantization range: lb={lb.max().item()}, ub={ub.min().item()}"

    if k == 1:
        quantizer = BinaryQuant.apply
        return quantizer(x, lb, ub)

    if align_zero:
        qx = dequantizer(linear_quant(x, lb, ub, k))
    else:
        # TODO: wrap in helper functions?
        round_ = RoundSTE.apply
        n = 2 ** k - 1
        delta = (ub - lb) / n
        x = clamp(x, lb, ub)
        qx = round_((x - lb) / delta) * delta + lb
    return qx
