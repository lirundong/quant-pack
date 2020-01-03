# -*- coding: utf-8 -*-

import torch
import torch.nn.functional as F

import quant_pack.operators as q_op

__all__ = ["fake_linear_quant", "quant_conv2d_forward", "quant_linear_forward"]


@torch.no_grad()
def prune_with_thresh_(x_f, x_q, prune_lb, prune_ub):
    # NOTE: this is only for evaluation
    if prune_lb is not None and prune_ub is not None:
        prune_mask = (prune_lb < x_f) & (x_f < prune_ub)
    elif prune_lb is not None:
        prune_mask = prune_lb < x_f
    else:
        prune_mask = x_f < prune_ub
    x_q.masked_fill_(prune_mask, 0.)


def fake_linear_quant(x, lb, ub, k, align_zero=False, prune_lb=None, prune_ub=None):
    if k == 32:
        return x
    elif k == 1:
        quantizer = q_op.BinaryFunc.apply
        qx = quantizer(x, lb, ub)
    else:
        quantizer = q_op.LinearQuantFunc.apply
        qx = quantizer(x, lb, ub, k, align_zero)
    if prune_lb is not None or prune_ub is not None:
        prune_with_thresh_(x, qx, prune_lb, prune_ub)
    return qx


def quant_conv2d_forward(module, input):
    raise NotImplementedError()


def quant_linear_forward(module, input):
    if module.input_transform is not None:
        input = module.input_transform(input)
    weight = module.weight
    if module.weight_transform is not None:
        weight = module.weight_transform(weight)
    output = F.linear(input, weight, module.bias)

    # TODO: remove the ad-hoc here
    if getattr(module, "gather_data", None):
        module.gather_buffer["input"] = input.detach()
        module.gather_buffer["weight"] = weight.detach()
        module.gather_buffer["output"] = output.detach()

    return output
