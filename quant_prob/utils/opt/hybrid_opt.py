# -*- coding: utf-8 -*-

from collections import OrderedDict
from typing import Tuple, Union
from math import sqrt
from itertools import chain

import torch
import torch.optim as optim

__all__ = ["HybridOpt"]

TensorT = torch.Tensor


class HybridOpt(object):

    def __init__(self,
                 param_groups,
                 conf_groups,
                 alter_step: int = 1):

        self.optimizers = []
        for group, conf in zip(param_groups, conf_groups):
            if len(group["params"]) > 0:
                opt = optim.__dict__[conf["type"]]([group, ], **conf["args"])
                self.optimizers.append(opt)

        self.alter_step = alter_step
        self.step_count = 0

    def state_dict(self):
        opt_dict = [("self", {k: v for k, v in self.__dict__.items() if "optimizers" not in k}), ]
        for i, opt in enumerate(self.optimizers):
            opt_dict.append((f"opt_{i}", opt.state_dict()))

        return OrderedDict(opt_dict)

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict["self"])
        for i, opt in enumerate(self.optimizers):
            opt.load_state_dict(state_dict[f"opt_{i}"])

    def get_param_groups(self):
        param_groups = []
        for group in chain(*(opt.param_groups for opt in self.optimizers)):
            param_groups.append(group)
        return param_groups

    def get_lr(self):
        lrs = []
        for group in self.get_param_groups():
            lrs.append(group["lr"])
        return lrs

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def zero_momentum(self):
        def _check_and_zero(state_dict, *keys):
            for key in keys:
                if key in state_dict:
                    state_dict[key].detach_()
                    state_dict[key].zero_()

        for opt in self.optimizers:
            if isinstance(opt, optim.SGD):
                for param_name, state in opt.state.items():
                    _check_and_zero(state, "momentum_buffer")
            elif isinstance(opt, optim.Adam):
                for param_name, state in opt.state.items():
                    state['step'] = 0
                    _check_and_zero(state, "exp_avg", "exp_avg_sq")

    def step(self, quant_enabled=False):
        self.step_count += 1
        if quant_enabled and len(self.optimizers) > 1:
            assert self.alter_step is not None
            enabled_idx = (self.step_count // self.alter_step) % len(self.optimizers)
            opt = self.optimizers[enabled_idx]
        else:
            opt = self.optimizers[0]
        opt.step()
