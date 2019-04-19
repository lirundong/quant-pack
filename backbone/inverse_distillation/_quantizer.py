# -*- coding: utf-8 -*-

from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.autograd import Function

__all__ = ["fake_quant"]

QuantT = Tuple[Tensor, Tensor, Tensor]  # I, z, delta


class RoundSTE(Function):

    @staticmethod
    def forward(ctx, x: Tensor) -> Tensor:
        return x.round()

    @staticmethod
    def backward(ctx, dy: Tensor) -> Tensor:
        return dy.clone()


def linear_quant(x: Tensor, lb: Tensor, ub: Tensor, k: int) -> QuantT:
    """ qx = (i - z) * delta

    Args:
        x:
        lb:
        ub:
        k:

    Returns:

    """
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


def fake_quant(x: Tensor, lb: Tensor, ub: Tensor, k: int) -> Tensor:
    qx = linear_quant(x, lb, ub, k)
    return dequantizer(qx)
