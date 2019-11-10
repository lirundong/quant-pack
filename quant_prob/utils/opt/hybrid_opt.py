# -*- coding: utf-8 -*-

import numbers
from collections import OrderedDict
from itertools import chain
from copy import copy

import torch
import torch.optim as optim

__all__ = ["HybridOpt"]

TensorT = torch.Tensor


class HybridOpt(object):

    @classmethod
    def _eval_lambda_literals(cls, cfg: dict):

        def _is_lambda_literal(literal):
            return isinstance(literal, str) and literal.startswith("lambda")

        def _check_and_convert_iterable(iterable):
            if isinstance(iterable, tuple):
                iterable = list(iterable)
            else:
                iterable = copy(iterable)
            for i, item in enumerate(iterable):
                if _is_lambda_literal(item):
                    iterable[i] = eval(item)
            return iterable

        for k, v in cfg.items():
            if _is_lambda_literal(v):
                _v = eval(v)
                cfg[k] = _v
            elif isinstance(v, (tuple, list)):
                _v = _check_and_convert_iterable(v)
                cfg[k] = _v
            elif isinstance(v, dict):
                _v = cls._eval_lambda_literals(v)
                cfg[k] = _v
        return cfg

    def __init__(self,
                 param_groups,
                 opt_cfgs,
                 alter_step=1):

        self.optimizers = []
        self.schedulers = []
        self.alter_step = alter_step

        for i, (param_group, opt_cfg) in enumerate(zip(param_groups, opt_cfgs)):
            if len(param_group["params"]) > 0:
                opt = optim.__dict__[opt_cfg["type"]]([param_group, ], **opt_cfg["args"])
                for j, schedule_cfg in enumerate(opt_cfg.get("schedules", [])):
                    schedule_name = schedule_cfg["name"]
                    schedule_args = self._eval_lambda_literals(schedule_cfg["args"])
                    scheduler = optim.lr_scheduler.__dict__[schedule_cfg["type"]](opt, **schedule_args)
                    self.schedulers.append((schedule_name, scheduler))
                self.optimizers.append(opt)

    def state_dict(self):
        """ Dump the internal states of registered optimizers.

        Note that states of related schedulers are manipulated by class `IterationScheduler`.
        """
        opt_dict = [("self", {k: v for k, v in self.__dict__.items() if
                              (not k.startswith("__")) and k != "optimizers" and k != "schedulers"}), ]
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

    def get_schedulers(self):
        return self.schedulers

    def zero_grad(self):
        for opt in self.optimizers:
            opt.zero_grad()

    def zero_momentum(self):

        def _check_and_zero(state_dict, *keys):
            for key in keys:
                if key in state_dict:
                    state_dict[key].detach_().zero_()

        for opt in self.optimizers:
            if isinstance(opt, optim.SGD):
                for param_name, state in opt.state.items():
                    _check_and_zero(state, "momentum_buffer")
            elif isinstance(opt, optim.Adam):
                for param_name, state in opt.state.items():
                    state["step"] = 0
                    _check_and_zero(state, "exp_avg", "exp_avg_sq")

    def step(self, current_iter, quant_enabled=False):
        if quant_enabled and len(self.optimizers) > 1:
            if self.alter_step is None:
                # take step on W and Theta jointly
                for opt in self.optimizers:
                    opt.step()
                return
            elif self.alter_step == -1:  # only optimize weights
                opt = self.optimizers[0]
            else:
                assert isinstance(self.alter_step, numbers.Integral)
                enabled_idx = (current_iter // self.alter_step) % len(self.optimizers)
                opt = self.optimizers[enabled_idx]
        else:
            opt = self.optimizers[0]
        opt.step()
