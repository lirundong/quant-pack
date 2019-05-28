# -*- coding: utf-8 -*-

from collections import OrderedDict
from typing import Tuple, Union
from math import sqrt

import torch
from torch.optim import Optimizer
import linklink as link

__all__ = ["QuantParamWeightOpt"]

TensorT = torch.Tensor


class QuantParamWeightOpt(object):

    eps = 1e-6

    def __init__(self,
                 weight_opt: Optimizer,
                 quant_param_opt: Optimizer,
                 soft_loss_weight: float = 0.25,
                 alter_step: int = 1,
                 warmup_step: int = -1,
                 world_size: int = 1,
                 debug: bool = False):
        assert 0 <= soft_loss_weight <= 1
        self.weight_opt = weight_opt
        self.quant_param_opt = quant_param_opt
        self.target_soft_loss_weight = soft_loss_weight
        self.target_hard_loss_weight = 1. - soft_loss_weight
        self.current_soft_loss_weight = self.target_soft_loss_weight
        self.current_hard_loss_weight = self.target_hard_loss_weight
        self.alter_step = alter_step
        self.warmup_step = warmup_step
        self.gone_step = 0
        self.world_size = world_size
        self.dist = world_size > 1
        self.debug = debug
        self.grad_source = "hard"  # `hard` or `soft`
        self.hard_grad_norm = 0.
        self.soft_grad_norm = 0.
        self.grad_handles = []
        self.register_grad_norm_hook()

    def register_grad_norm_hook(self):

        def grad_norm_acc_hook(grad: TensorT) -> Union[TensorT, None]:
            """Record gradient norms and scale with hard/soft loss weights."""
            if not self.debug:
                return
            if self.grad_source == "hard":
                self.hard_grad_norm += grad.data.norm().item() ** 2
                return grad.mul(self.current_hard_loss_weight)
            elif self.grad_source == "soft":
                self.soft_grad_norm += grad.data.norm().item() ** 2
                return grad.mul(self.current_soft_loss_weight)
            else:
                raise ValueError(f"unknown grad source: {self.grad_source}")

        if self.debug:
            weight_param = set()
            quant_param = set()
            for group in self.weight_opt.param_groups:
                weight_param.update(group["params"])
            for group in self.quant_param_opt.param_groups:
                quant_param.update(group["params"])
            assert len(weight_param.intersection(quant_param)) == 0

            assert len(self.grad_handles) == 0
            for p in weight_param:
                h = p.register_hook(grad_norm_acc_hook)
                self.grad_handles.append(h)

    def remove_grad_norm_hook(self):
        assert len(self.grad_handles) > 0
        for h in self.grad_handles:
            h.remove()

    def enabled_optimizer(self, step: int, quant_enabled: bool = False) -> Optimizer:
        if quant_enabled:
            div = step // self.alter_step
            if div % 2 == 0:
                return self.weight_opt
            else:
                return self.quant_param_opt
        else:
            return self.weight_opt

    def zero_grad(self):
        self.weight_opt.zero_grad()
        self.quant_param_opt.zero_grad()
        self.hard_grad_norm = 0.
        self.soft_grad_norm = 0.

    def state_dict(self):
        return OrderedDict([("weight_opt", self.weight_opt.state_dict()),
                            ("quant_param_opt", self.quant_param_opt.state_dict()),
                            ("self", {k: v for k, v in self.__dict__.items() if "opt" not in k})])

    def load_state_dict(self, state_dict):
        self.weight_opt.load_state_dict(state_dict["weight_opt"])
        self.quant_param_opt.load_state_dict(state_dict["quant_param_opt"])
        self.__dict__.update(state_dict["self"])

    def get_lr(self):
        weight_lr = self.weight_opt.param_groups[0]["lr"]
        quant_param_lr = self.quant_param_opt.param_groups[0]["lr"]
        return weight_lr, quant_param_lr

    def step(self):
        # TODO: loss weight schedule here
        self.gone_step += 1
        if self.warmup_step <= 0 or self.warmup_step < self.gone_step:
            self.current_soft_loss_weight = self.target_soft_loss_weight
            self.current_hard_loss_weight = self.target_hard_loss_weight
        else:
            soft_weight_delta = self.target_soft_loss_weight / self.warmup_step
            soft_loss_weight = soft_weight_delta * self.gone_step
            hard_loss_weight = 1. - soft_loss_weight
            self.current_soft_loss_weight = soft_loss_weight
            self.current_hard_loss_weight = hard_loss_weight

    def backward(self,
                 hard_loss: TensorT,
                 soft_loss: TensorT,
                 get_grad_norm: bool = False) -> Union[float, Tuple[float, float, float]]:
        self.zero_grad()
        self.step()
        if get_grad_norm:
            assert len(self.grad_handles) > 0
            self.debug = True
            self.grad_source = "hard"
            hard_loss.div(self.world_size).backward(retain_graph=True)
            if self.dist:
                link.synchronize()
            self.grad_source = "soft"
            soft_loss.div(self.world_size).backward()
            if self.dist:
                link.synchronize()
            total_loss = soft_loss.item() + hard_loss.item()
            return total_loss, sqrt(self.hard_grad_norm + self.eps), sqrt(self.soft_grad_norm + self.eps)
        else:
            self.debug = False
            total_loss = hard_loss * self.current_hard_loss_weight + soft_loss * self.current_soft_loss_weight
            total_loss.div(self.world_size).backward()
            if self.dist:
                link.synchronize()
            return total_loss.item()

