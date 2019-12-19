# -*- coding: utf-8 -*-

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import Parameter
from torch.distributions import Multinomial
from torchvision.utils import make_grid


class Visible:
    def __init__(self, vis_channels=None):
        self.vis = False
        self.name = None
        self.iter = -1
        self.tb_logger = None
        self.vis_channels = vis_channels

    def vis_feat(self, feat):
        with torch.no_grad():
            prefix = self.name if self.training else f"{self.name}/eval"
            self.tb_logger.add_histogram(prefix + "/feat", feat, self.iter)

            if isinstance(self, nn.Conv2d):
                y0 = feat[0].unsqueeze(0).permute(1, 0, 2, 3)
                if self.vis_channels is not None:
                    y0 = y0[self.vis_channels]
                n_rows = math.ceil(math.sqrt(y0.size(0)))
                y0_img = make_grid(y0, nrow=n_rows, normalize=True)
                self.tb_logger.add_image(prefix + "/feat", y0_img, self.iter)

    def vis_error(self, feat, ref_feat):
        with torch.no_grad():
            err = feat - ref_feat
            snr = 10. * torch.log10(ref_feat.norm() / err.norm())
            prefix = self.name if self.training else f"{self.name}/eval"
            self.tb_logger.add_histogram(prefix + "/err", err, self.iter)
            self.tb_logger.add_scalar("SNR/" + prefix, snr.item(), self.iter)

    def vis_weight(self):
        if self.vis and self.iter <= 1:
            w = self.weight.detach()
            self.tb_logger.add_histogram(f"{self.name}/weight", w.view(-1), self.iter)

    def vis_bound(self):
        if self.vis:
            self.tb_logger.add_scalar(f"{self.name}/lb", self.lb.item(), self.iter)
            self.tb_logger.add_scalar(f"{self.name}/ub", self.ub.item(), self.iter)

    def forward_vis(self, y):
        # TODO: rewrite as decorator
        if self.vis:
            self.vis_feat(y)
            if self.is_teacher:
                for m in self.students:
                    m.ref_feat = y.data
            else:
                # import pdb; pdb.set_trace()
                if self.ref_feat is not None:
                    assert self.ref_feat.shape == y.shape, \
                        f"FP and quant feat on layer `{self.name}` shape not match"
                    self.vis_error(y, self.ref_feat)
                    self.ref_feat = None
            self.vis = False


###############################################################################
# Denoiser
###############################################################################


class NonLocal(nn.Module):

    def __init__(self, channels, inplace=False, softmax=False):
        super().__init__()
        self.channels = channels
        self.inplace = inplace
        self.softmax = softmax
        self.mapping = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

    def forward(self, x):
        n, c, h, w = x.shape
        res = x

        # adapted from: https://github.com/facebookresearch/ImageNet-Adversarial-Training
        theta, phi, g = x, x, x
        if c > h * w or self.softmax:
            f = torch.einsum('niab,nicd->nabcd', theta, phi)
            if self.softmax:
                orig_shape = f.shape
                f = f.reshape(-1, h * w, h * w)
                f = f / torch.sqrt(torch.tensor(c, device=f.device, dtype=f.dtype))
                f = torch.softmax(f)
                f = f.reshape(orig_shape)
            f = torch.einsum('nabcd,nicd->niab', f, g)
        else:
            f = torch.einsum('nihw,njhw->nij', phi, g)
            f = torch.einsum('nij,nihw->njhw', f, theta)
        if not self.softmax:
            f = f / torch.tensor(h * w, device=f.device, dtype=f.dtype)
        f = f.reshape(x.shape)

        y = self.mapping(f) + res
        y = F.relu(y, self.inplace)

        return y


###############################################################################
# Naive quant with STE
###############################################################################


class NaiveQuantSTE(Function):

    @staticmethod
    def forward(ctx, x, k):
        x = x.clone()
        n = 2 ** k - 1
        lb = x.min()
        delta = x.max().sub_(lb).div_(n)
        return x.sub(lb).div_(delta).round_().mul_(delta).add_(lb)

    @staticmethod
    def backward(ctx, dy):
        return dy.clone(), None


class NaiveQuantConv2d(nn.Conv2d, Visible):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 quant=False, bit_width=4, vis_channels=None):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias)
        Visible.__init__(self, vis_channels)
        self.quant = quant
        self.bit_width = bit_width
        self.register_buffer("ref_feat", None)

    def forward(self, input):
        if self.quant:
            q = NaiveQuantSTE.apply
            w = q(self.weight, self.bit_width)
            y = F.conv2d(input, w, self.bias, self.stride,
                         self.padding, self.dilation, self.groups)
        else:
            y = super().forward(input)

        self.forward_vis(y)
        return y


class NaiveQuantLinear(nn.Linear, Visible):

    def __init__(self, in_features, out_features, bias=True, quant=False,
                 bit_width=4, vis_channels=None):
        super().__init__(in_features, out_features, bias)
        Visible.__init__(self, vis_channels)
        self.quant = quant
        self.bit_width = bit_width
        self.register_buffer("ref_feat", None)

    def forward(self, input):
        if self.quant:
            q = NaiveQuantSTE.apply
            w = q(self.weight, self.bit_width)
            y = F.linear(input, w, self.bias)
        else:
            y = super().forward(input)

        self.forward_vis(y)
        return y

###############################################################################
# Differentiable quant boundaries
###############################################################################


class RoundSTE(Function):

    @staticmethod
    def forward(ctx, x):
        return x.round()

    @staticmethod
    def backward(ctx, dy):
        return dy.clone()


class DiffBoundary:

    def __init__(self, bit_width=4):
        # TODO: add channel-wise option?
        self.bit_width = bit_width
        self.register_boundaries()

    def register_boundaries(self):
        assert hasattr(self, "weight")
        self.lb = Parameter(self.weight.data.min())
        self.ub = Parameter(self.weight.data.max())

    def reset_boundaries(self):
        assert hasattr(self, "weight")
        self.lb.data = self.weight.data.min()
        self.ub.data = self.weight.data.max()

    def get_quant_weight(self, align_zero=True):
        # TODO: set `align_zero`?
        if align_zero:
            return self._get_quant_weight_align_zero()
        else:
            return self._get_quant_weight()

    def _get_quant_weight(self):
        round_ = RoundSTE.apply
        w = self.weight.detach()
        delta = (self.ub - self.lb) / (2 ** self.bit_width - 1)
        w = torch.clamp(w, self.lb.item(), self.ub.item())
        idx = round_((w - self.lb).div(delta))  # TODO: do we need STE here?
        qw = (idx * delta) + self.lb
        return qw

    def _get_quant_weight_align_zero(self):
        # TODO: WTF?
        round_ = RoundSTE.apply
        n = 2 ** self.bit_width - 1
        w = self.weight.detach()
        delta = (self.ub - self.lb) / n
        z = round_(self.lb.abs() / delta)
        lb = -z * delta
        ub = (n - z) * delta
        w = torch.clamp(w, lb.item(), ub.item())
        idx = round_((w - self.lb).div(delta))  # TODO: do we need STE here?
        qw = (idx - z) * delta
        return qw


class QConv2dDiffBounds(nn.Conv2d, DiffBoundary, Visible):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 quant=False, bit_width=4):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias)
        self.quant = quant
        DiffBoundary.__init__(self, bit_width)
        Visible.__init__(self)

    def forward(self, input):
        if self.quant:
            w = self.get_quant_weight()
            y = F.conv2d(input, w, self.bias, self.stride,
                         self.padding, self.dilation, self.groups)
        else:
            y = super().forward(input)

        self.vis_weight()
        self.vis_bound()
        return y


class QLinearDiffBounds(nn.Linear, DiffBoundary, Visible):

    def __init__(self, in_features, out_features, bias=True, quant=False,
                 bit_width=4):
        super().__init__(in_features, out_features, bias)
        self.quant = quant
        DiffBoundary.__init__(self, bit_width)
        Visible.__init__(self)

    def forward(self, input):
        if self.quant:
            w = self.get_quant_weight()
            y = F.linear(input, w, self.bias)
        else:
            y = super().forward(input)

        self.vis_weight()
        self.vis_bound()
        return y

###############################################################################
# Probabilistic quant with local reparameterization trick
###############################################################################


def inv_sigmoid(x, lb, ub):
    x = torch.clamp(x, lb, ub)
    return - torch.log(1. / x - 1.)


class TernaryConv2d(nn.Conv2d, Visible):
    """Implementation of `LR-Nets`(https://arxiv.org/abs/1710.07739)."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, quant=False,
                 p_max=0.95, p_min=0.05, vis_channels=None):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation, groups, bias)
        Visible.__init__(self, vis_channels)

        self.quant = quant
        self.p_max = p_max
        self.p_min = p_min
        self.eps = 1e-5
        self.register_buffer("w_candidate", torch.tensor([-1., 0., 1.]))
        self.p_a = Parameter(torch.zeros_like(self.weight))
        self.p_b = Parameter(torch.zeros_like(self.weight))
        self.reset_p()

    def reset_p(self):
        w = self.weight.data / self.weight.data.std()
        self.p_a.data = inv_sigmoid(self.p_max - (self.p_max - self.p_min) * w.abs(), self.p_min, self.p_max)
        self.p_b.data = inv_sigmoid(0.5 * (1. + w / (1. - torch.sigmoid(self.p_a.data))), self.p_min, self.p_max)

    def forward(self, input):
        if self.quant:
            p_a = torch.sigmoid(self.p_a)
            p_b = torch.sigmoid(self.p_b)
            p_w_0 = p_a
            p_w_pos = p_b * (1. - p_w_0)
            p_w_neg = (1. - p_b) * (1. - p_w_0)
            p = torch.stack([p_w_neg, p_w_0, p_w_pos], dim=-1)
            if self.training:
                w_mean = (p * self.w_candidate).sum(dim=-1)
                w_var = (p * self.w_candidate.pow(2)).sum(dim=-1) - w_mean.pow(2)
                act_mean = F.conv2d(input, w_mean, self.bias, self.stride,
                                    self.padding, self.dilation, self.groups)
                act_var = F.conv2d(input.pow(2), w_var, None, self.stride,
                                   self.padding, self.dilation, self.groups)
                var_eps = torch.randn_like(act_mean)
                y = act_mean + var_eps * act_var.add(self.eps).sqrt()
            else:
                m = Multinomial(probs=p)
                indices = m.sample().argmax(dim=-1)
                w = self.w_candidate[indices]
                y = F.conv2d(input, w, self.bias, self.stride,
                             self.padding, self.dilation, self.groups)
        else:
            y = super().forward(input)

        self.forward_vis(y)
        return y


class TernaryLinear(nn.Linear, Visible):
    """Implementation of `LR-Nets`(https://arxiv.org/abs/1710.07739)."""

    def __init__(self, in_features, out_features, bias=True, quant=False,
                 p_max=0.95, p_min=0.05, vis_channels=None):
        super().__init__(in_features, out_features, bias)
        Visible.__init__(self, vis_channels)

        self.quant = quant
        self.p_max = p_max
        self.p_min = p_min
        self.eps = 1e-5
        self.register_buffer("w_candidate", torch.tensor([-1., 0., 1.]))
        self.p_a = Parameter(torch.zeros_like(self.weight))
        self.p_b = Parameter(torch.zeros_like(self.weight))
        self.reset_p()

    def reset_p(self):
        w = self.weight.data / self.weight.data.std()
        self.p_a.data = inv_sigmoid(self.p_max - (self.p_max - self.p_min) * w.abs(), self.p_min, self.p_max)
        self.p_b.data = inv_sigmoid(0.5 * (1. + w / (1. - torch.sigmoid(self.p_a.data))), self.p_min, self.p_max)

    def forward(self, input):
        if self.quant:
            p_a = torch.sigmoid(self.p_a)
            p_b = torch.sigmoid(self.p_b)
            p_w_0 = p_a
            p_w_pos = p_b * (1. - p_w_0)
            p_w_neg = (1. - p_b) * (1. - p_w_0)
            p = torch.stack([p_w_neg, p_w_0, p_w_pos], dim=-1)
            if self.training:
                w_mean = (p * self.w_candidate).sum(dim=-1)
                w_var = (p * self.w_candidate.pow(2)).sum(dim=-1) - w_mean.pow(2)
                act_mean = F.linear(input, w_mean, self.bias)
                act_var = F.linear(input.pow(2), w_var, None)
                var_eps = torch.randn_like(act_mean)
                y = act_mean + var_eps * act_var.add(self.eps).sqrt()
            else:
                m = Multinomial(probs=p)
                indices = m.sample().argmax(dim=-1)
                w = self.w_candidate[indices]
                y = F.linear(input, w, self.bias)
        else:
            y = super().forward(input)

        self.forward(y)
        return y
