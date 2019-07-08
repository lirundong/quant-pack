# -*- coding: utf-8 -*-

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function

__all__ = ["fake_quant", "clamp"]

QuantT = Tuple[Tensor, Tensor, Tensor]  # I, z, delta


class RoundSTE(Function):

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        return x.round()

    @staticmethod
    def backward(ctx, dy: Tensor) -> Tensor:
        return dy.clone()


def clamp(x: Tensor, lb: Tensor, ub: Tensor) -> Tensor:
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
    with torch.no_grad():
        lb = z.neg().mul(delta)
        ub = (n - z).mul(delta)
    x = torch.clamp(x, lb.item(), ub.item())
    i = round_(x.sub(lb).div(delta))

    return i, z, delta


def dequantizer(qx: QuantT) -> Tensor:
    i, z, delta = qx
    return i.sub(z).mul(delta)


def fake_quant(x: Tensor, lb: Tensor, ub: Tensor, k: int, align_zero: bool = True) -> Tensor:
    assert lb.lt(ub).all(), f"invalid quantization range: lb={lb.max().item()}, ub={ub.min().item()}"
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
