# -*- coding: utf-8 -*-

import math

from quant_pack.core.quant.functional import fake_linear_quant

_registered_quantizers = {
    "linear": fake_linear_quant,
}

__all__ = ["QuantConfig"]


class QuantConfig:

    def __init__(self, mode, bit_width, lb, ub, align_zero, prune_to_zero=False):
        self.mode = mode
        self.bit_width = bit_width
        self.lb = lb
        self.ub = ub
        self.align_zero = align_zero
        self.prune_to_zero = prune_to_zero
        self.retain_fp = False

        self._enabled = True
        self._quantizer = _registered_quantizers[self.mode]

    def quant(self, enabled=True):
        self._enabled = enabled

    def fc(self):
        self.quant(enabled=False)

    @property
    def transform(self):
        if self._enabled and not self.retain_fp:
            if self.prune_to_zero:
                lb, ub = self.lb.item(), self.ub.item()
                delta = (ub - lb) / (2 ** self.bit_width - 1)
                if 0. < lb:
                    p_lb = None
                    p_ub = lb - delta / 2
                elif ub < 0.:
                    p_lb = ub + delta / 2
                    p_ub = None
                else:
                    q0_minor = lb + math.floor(abs(lb) / delta) * delta
                    q0_plus = lb + math.ceil(abs(lb) / delta) * delta
                    p_lb = q0_minor / 2
                    p_ub = q0_plus / 2
            else:
                p_lb = p_ub = None
            return lambda x: self._quantizer(x, self.lb, self.ub, self.bit_width, self.align_zero, p_lb, p_ub)
        else:
            return None

    @property
    def params(self):
        if self.align_zero:
            # return scale (float), zero_point (int), dtype (torch.uint8 / torch.int8)
            raise NotImplementedError()
        else:
            lb, ub = self.lb.item(), self.ub.item()
            delta = (ub - lb) / (2 ** self.bit_width - 1)
            return lb, ub, delta

    @property
    def enabled(self):
        return self._enabled
