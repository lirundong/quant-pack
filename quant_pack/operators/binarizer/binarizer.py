# -*- coding: utf-8 -*-

import torch
from torch.autograd import Function

from quant_pack.operators.binarizer._C import binary_forward, binary_backward


class BinaryFunc(Function):

    @staticmethod
    def forward(ctx, x, lb, ub):
        with torch.no_grad():
            assert lb.lt(ub), f"invalid binarization range: lb={lb.max().item()}, ub={ub.min().item()}"
        x = x.contiguous()
        qx, mask_x = binary_forward(x, lb, ub)
        ctx.save_for_backward(mask_x)
        return qx

    @staticmethod
    def backward(ctx, dy):
        dy = dy.clone()
        mask_x = ctx.saved_tensors
        dx, dlb, dub = binary_backward(dy, mask_x)
        return dx, dlb, dub
