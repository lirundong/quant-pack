# -*- coding: utf-8 -*-

import math
from enum import Flag, auto

from quant_pack.core.quant.functional import fake_linear_quant

_registered_quantizers = {
    "linear": fake_linear_quant,
}

__all__ = ["QuantConfig", "QuantMode"]


class _SequentialLambdas:

    def __init__(self, *args):
        self.ops = []
        for arg in args:
            if arg is not None:
                self.ops.append(arg)

    def __call__(self, input):
        for op in self.ops:
            input = op(input)
        return input


def combine_optional_callables(*args):
    if all(arg is None for arg in args):
        return None
    else:
        return _SequentialLambdas(*args)


class QuantMode(Flag):
    QW = auto()
    QA = auto()
    FW = ~QW
    FA = ~QA
    QWQA = QW | QA
    FWFA = FW | FA
    QWFA = QW | FA
    FWQA = FW | QA

    # BC: build QuantMode from VALID_QUANT_MODE in qat_policies
    @classmethod
    def get(cls, mode):
        assert isinstance(mode, str)
        if mode == "quant":
            return cls.QWQA
        elif mode == "fp":
            return cls.FWFA
        elif mode == "qw_fa":
            return cls.QWFA
        elif mode == "fw_qa":
            return cls.FWQA

    def __str__(self):
        if self is QuantMode.FWFA:
            return "fp"
        elif self is QuantMode.QWQA:
            return "quant"
        elif self is QuantMode.QWFA:
            return "qw_fa"
        elif self is QuantMode.FWQA:
            return "fw_qa"
        else:
            return super(QuantMode, self).__str__()


class QuantConfig:

    def __init__(self, method, bit_width, lb, ub, align_zero, prune_to_zero=False):
        self.method = method
        self.bit_width = bit_width
        self.lb = lb
        self.ub = ub
        self.align_zero = align_zero
        self.prune_to_zero = prune_to_zero
        self.retain_fp = False

        self._enabled = True
        self._quantizer = _registered_quantizers[self.method]
        self._manual_bias = None  # experimental

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
            q_f = lambda x: self._quantizer(x, self.lb, self.ub, self.bit_width, self.align_zero, p_lb, p_ub)
        else:
            q_f = None

        if self._manual_bias is not None:
            bias_f = lambda x: x + x.detach().mul_(self._manual_bias)
        else:
            bias_f = None

        return combine_optional_callables(q_f, bias_f)

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
