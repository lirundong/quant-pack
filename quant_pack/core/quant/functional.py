# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

import quant_pack.operators as q_op

__all__ = ["fake_linear_quant", "quant_conv2d_forward", "quant_linear_forward"]


def fake_linear_quant(x, lb, ub, k, align_zero=False):
    if k == 32:
        return x
    elif k == 1:
        quantizer = q_op.BinaryFunc.apply
        return quantizer(x, lb, ub)
    else:
        quantizer = q_op.LinearQuantFunc.apply
        return quantizer(x, lb, ub, k, align_zero)


def quant_conv2d_forward(module, input):
    raise NotImplementedError()


def quant_linear_forward(module, input):
    raise NotImplementedError()
