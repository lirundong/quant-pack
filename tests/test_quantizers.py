# -*- coding: utf-8 -*-

import torch
from torch.nn import Parameter

from quant_prob.modeling.quantizers.param_linear_quantizer import fake_linear_quant, clamp
from quant_prob.modeling.quantizers.cuda_param_linear_quantizer import cuda_fake_linear_quant

SEED = 19260817
DEVICE = torch.device("cuda:0")
DTYPE = torch.float64
torch.manual_seed(SEED)


def test_bound_grad():
    x = torch.randn(1, 3, 224, 224, requires_grad=True)
    lb = Parameter(x.detach().min() + 0.1)
    ub = Parameter(x.detach().max() - 0.1)
    k = 8

    # shrink boundaries manually
    assert lb.requires_grad
    assert ub.requires_grad

    qx = fake_linear_quant(x, lb, ub, k, align_zero=True)
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


def test_quant_num_grad_align_zero():
    # TODO: we should add gradients to `clamp` op here
    x = torch.randn(1, 3, 224, 224, requires_grad=True, dtype=DTYPE, device=DEVICE)
    d_qx = torch.randn_like(x).detach()
    lb = Parameter(x.detach().min() + 0.1)
    ub = Parameter(x.detach().max() - 0.1)
    k = 8

    # autograd implementation
    assert ub.detach() - lb.detach() > 1e-2
    qx = fake_linear_quant(x, lb, ub, k, align_zero=True)
    qx.backward(d_qx)

    qx_gt = qx.detach()
    d_lb_gt = lb.grad.detach()
    d_ub_gt = ub.grad.detach()
    d_x_gt = x.grad.detach()

    # CUDA numerical implementation
    lb.grad.data.zero_()
    ub.grad.data.zero_()
    x.grad.data.zero_()

    qx = cuda_fake_linear_quant(x, lb, ub, k, align_zero=True)
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
        N = torch.tensor(2 ** k - 1, dtype=DTYPE, device=DEVICE)
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


def test_quant_num_grad_no_align_zero():
    from quant_prob.modeling.quantizers.param_linear_quantizer import RoundSTE
    x = torch.randn(1, 3, 224, 224, requires_grad=True, dtype=DTYPE, device=DEVICE)
    d_qx = torch.randn_like(x).detach()
    lb = Parameter(x.detach().min() + 0.1)
    ub = Parameter(x.detach().max() - 0.1)
    k = 8
    round_ = RoundSTE.apply

    # watch internal grad
    grad_acc = torch.zeros_like(lb, requires_grad=False)
    grad_n = torch.tensor(0)

    def get_internal_grad(grad):
        grad_acc.add_(grad)
        grad_n.add_(1)

    # autograd implementation
    assert ub.detach() - lb.detach() > 1e-2
    n = 2 ** k - 1
    delta = (ub - lb) / n
    x_clamped = clamp(x, lb, ub)
    qx = round_((x_clamped - lb) / delta) * delta + lb
    with torch.no_grad():
        qi_diff = round_((x_clamped - lb) / delta) - ((x_clamped - lb) / delta)
        x_mask = (lb <= x) & (x <= ub)  # pre-compute mask
        mask_up = (x > ub).to(x.dtype)
        mask_down = (x < lb).to(x.dtype)
        # NOTE: the diff_i matters!
        d_ub = (d_qx * mask_up).sum() + (d_qx * qi_diff / n).sum()
        d_lb = (d_qx * mask_down).sum() - (d_qx * qi_diff / n).sum()
    delta.register_hook(get_internal_grad)
    qx.backward(d_qx)

    d_lb_gt = lb.grad.detach()
    d_ub_gt = ub.grad.detach()
    d_x_gt = x.grad.detach()

    assert torch.allclose(d_qx * x_mask.to(d_qx.dtype), d_x_gt)
    assert torch.isclose(d_ub, d_ub_gt)
    assert torch.isclose(d_lb, d_lb_gt)

    # CUDA numerical implementation
    lb.grad.data.zero_()
    ub.grad.data.zero_()
    x.grad.data.zero_()

    qx = cuda_fake_linear_quant(x, lb, ub, k, align_zero=False)
    qx.backward(d_qx)

    d_lb_cuda = lb.grad.detach()
    d_ub_cuda = ub.grad.detach()
    d_x_cuda = x.grad.detach()

    assert torch.allclose(d_lb_cuda, d_lb_gt)
    assert torch.allclose(d_ub_cuda, d_ub_gt)
    assert torch.allclose(d_x_cuda, d_x_gt)


def test_clamp_num_grad():
    x = torch.randn(1, 3, 224, 224, requires_grad=True, dtype=DTYPE, device=DEVICE)
    d_qx = torch.randn_like(x).detach()
    lb = Parameter(x.detach().min() + 0.1)
    ub = Parameter(x.detach().max() - 0.1)

    x_clamped = clamp(x, lb, ub)
    x_clamped.backward(d_qx)
    with torch.no_grad():
        mask_up = (x > ub).to(x.dtype)
        mask_down = (x < lb).to(x.dtype)
        d_ub = (d_qx * mask_up).sum()
        d_lb = (d_qx * mask_down).sum()

    assert torch.isclose(d_ub, ub.grad.detach())
    assert torch.isclose(d_lb, lb.grad.detach())
