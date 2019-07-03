# -*- coding: utf-8 -*-

import pytest
import torch
from torch.nn import Parameter

from backbone.inverse_distillation._quantizer import linear_quant, dequantizer, fake_quant
from utils.quant.linear_quantizer import fake_linear_quant

SEED = 19260817
torch.manual_seed(SEED)


def test_bound_grad():
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    lb = Parameter(x.detach().min() + 0.1)
    ub = Parameter(x.detach().max() - 0.1)
    k = 8

    # shrink boundaries manually
    assert lb.requires_grad
    assert ub.requires_grad

    qx = fake_quant(x, lb, ub, k, align_zero=True)
    d_qx = torch.randn_like(qx)
    qx.backward(d_qx)

    assert lb.grad is not None
    assert ub.grad is not None


@torch.no_grad()
def d_lb_ub(dy, d_i, N, sign_lb):
    d_delta = dy * d_i
    d_ub = d_delta / N
    d_lb = - d_ub - dy * sign_lb
    return d_lb.sum(), d_ub.sum()


@torch.no_grad()
def d_x(dy, mask_x):
    return dy.clone() * mask_x.to(dy.dtype)


def test_quant_num_grad():
    device = torch.device("cuda:0")
    dtype = torch.float64
    x = torch.randn(1, 3, 224, 224, requires_grad=True, dtype=dtype, device=device)
    d_qx = torch.randn_like(x).detach()
    lb = Parameter(x.detach().min() + 0.1)
    ub = Parameter(x.detach().max() - 0.1)
    k = 8

    # autograd implementation
    assert ub.detach() - lb.detach() > 1e-2
    qx = fake_quant(x, lb, ub, k, align_zero=True)
    qx.backward(d_qx)

    qx_gt = qx.detach()
    d_lb_gt = lb.grad.detach()
    d_ub_gt = ub.grad.detach()
    d_x_gt = x.grad.detach()

    # CUDA numerical implementation
    lb.grad.data.zero_()
    ub.grad.data.zero_()
    x.grad.data.zero_()

    qx = fake_linear_quant(x, lb, ub, k, align_zero=True)
    qx.backward(d_qx)

    qx_cuda = qx.detach()
    d_lb_cuda = lb.grad.detach()
    d_ub_cuda = ub.grad.detach()
    d_x_cuda = x.grad.detach()

    assert torch.allclose(qx_cuda, qx_gt)
    assert torch.allclose(d_lb_cuda, d_lb_gt)
    assert torch.allclose(d_ub_cuda, d_ub_gt)
    assert torch.allclose(d_x_cuda, d_x_gt)

    # numerical grad implementation
    with torch.no_grad():
        N = torch.tensor(2 ** k - 1, dtype=dtype, device=device)
        delta = ub.sub(lb).div(N)
        z = torch.round(lb.abs().div(delta))
        lb_ = z.neg().mul(delta)
        ub_ = (N - z).mul(delta)
        x_mask = (lb_ <= x) & (x <= ub_)  # pre-compute mask
        x = torch.clamp(x, lb_.item(), ub_.item())
        i = torch.round(x.sub(lb_).div(delta))

        # after forward, calculate cache
        x_sub = x - lb_ - torch.abs(lb)
        d_i = (i - z) - (x_sub / delta)
        d_lb, d_ub = d_lb_ub(d_qx, d_i, N, torch.sign(lb))
        dx = d_x(d_qx, x_mask)

        assert torch.allclose(d_lb_gt, d_lb)
        assert torch.allclose(d_ub_gt, d_ub)
        assert torch.allclose(dx, d_x_gt)
