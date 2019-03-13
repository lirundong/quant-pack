# -*- coding: utf-8 -*-

from copy import copy

import torch
import torch.nn as nn

from ._components import TernaryConv2d, TernaryLinear, NaiveQuantConv2d, \
                         NaiveQuantLinear, NonLocal

__all__ = ["QuantBackbone"]

quantifiable = (TernaryConv2d, TernaryLinear, NaiveQuantConv2d, NaiveQuantLinear)
visible = (TernaryConv2d, TernaryLinear, NaiveQuantConv2d, NaiveQuantLinear)
probabilistic = (TernaryConv2d, TernaryLinear)
decayable = (nn.Conv2d, nn.Linear)
param_denoise = (NonLocal, )


class QuantBackbone(nn.Module):

    def opt_param_groups(self, opt_prob=False, denoise_only=False, **opt_conf):
        decay_group = dict(params=[], **opt_conf)
        no_decay_conf = copy(opt_conf)
        no_decay_conf['weight_decay'] = 0.
        no_decay_group = dict(params=[], **no_decay_conf)

        if denoise_only:
            for m in self.modules():
                if isinstance(m, param_denoise):
                    for p in m.parameters():
                        no_decay_group["params"].append(p)
            return [no_decay_group, ]

        def param_filter(name, module):
            if isinstance(module, probabilistic) and any(s in name for s in ("weight", "p_a", "p_b")):
                if opt_prob:
                    return any(s in name for s in ("p_a", "p_b"))
                else:
                    return "weight" in name
            else:
                return torch.is_tensor(p)

        for m in self.modules():
            for n, p in m._parameters.items():
                if param_filter(n, m):
                    if isinstance(m, decayable) and any(s in n for s in ("weight", "p_a", "p_b")):
                        decay_group["params"].append(p)
                    else:
                        no_decay_group["params"].append(p)

        return [decay_group, no_decay_group]

    def quant(self, enable=True):
        for m in self.modules():
            if isinstance(m, quantifiable):
                m.quant = enable

    def full_precision(self):
        self.quant(False)

    def reset_p(self):
        for m in self.modules():
            if isinstance(m, probabilistic):
                m.reset_p()

    def register_vis(self, tb_logger, sub_fix=None):
        for n, m in self.named_modules():
            if isinstance(m, visible):
                if sub_fix is not None:
                    n = f"{n}.{sub_fix}"
                m.name = n
                m.tb_logger = tb_logger
                if not hasattr(m, "is_teacher"):
                    m.is_teacher = False

    def vis(self, iter):
        for n, m in self.named_modules():
            if isinstance(m, visible):
                m.iter = iter
                m.vis = True

    def register_teacher(self, teacher):
        """Very toxic, take care when playing."""
        assert self.__class__ == teacher.__class__, f"distillation pairs should come from same class"
        for (n1, m1), (n2, m2) in zip(self.named_modules(), teacher.named_modules()):
            if isinstance(m1, visible) and isinstance(m2, visible):
                assert n1 == n2
                m1.is_teacher = False
                m2.is_teacher = True
                if not hasattr(m2, "students"):
                    m2.students = [m1, ]  # do not directly assign modules!
                else:
                    m2.students.append(m1)
