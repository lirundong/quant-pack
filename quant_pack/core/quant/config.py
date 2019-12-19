# -*- coding: utf-8 -*-

from quant_pack.core.quant.functional import fake_linear_quant

_registered_quantizers = {
    "linear": fake_linear_quant,
}

__all__ = ["QuantConfig"]


class QuantConfig:

    def __init__(self, mode, bit_width, lb, ub, align_zero):
        self.mode = mode
        self.bit_width = bit_width
        self.lb = lb
        self.ub = ub
        self.align_zero = align_zero

        self._enabled = True
        self._quantizer = _registered_quantizers[self.mode]

    def quant(self, enabled=True):
        self._enabled = enabled

    def fc(self):
        self.quant(enabled=False)

    @property
    def transform(self):
        if self._enabled:
            return lambda x: self._quantizer(x, self.lb, self.ub, self.bit_width, self.align_zero)
        else:
            return None
