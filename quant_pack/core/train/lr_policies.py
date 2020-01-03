# -*- coding: utf-8 -*-

import torch.distributed as dist
from torch.optim.optimizer import Optimizer
from mmcv.runner import LrUpdaterHook


def _get_applied_param_groups(runner, apply_to):
    if isinstance(runner.optimizer, Optimizer):
        return runner.optimizer.param_groups
    param_groups = []
    for optim in (runner.optimizer[name] for name in apply_to):
        param_groups += optim.param_groups
    return param_groups


class _MultiOptimLrUpdateHook(LrUpdaterHook):

    def __init__(self, apply_to, scale_by_world_size=False, by_epoch=True,
                 warmup=None, warmup_iters=0, warmup_ratio=0.1):
        super(_MultiOptimLrUpdateHook, self).__init__(by_epoch, warmup, warmup_iters, warmup_ratio)
        assert isinstance(apply_to, (list, tuple)) and len(apply_to) > 0
        self.apply_to = apply_to
        self.scale_by_world_size = scale_by_world_size

    def _set_lr(self, runner, lr_groups):
        param_groups = _get_applied_param_groups(runner, self.apply_to)
        assert len(param_groups) == len(lr_groups)
        for param_group, lr in zip(param_groups, lr_groups):
            param_group["lr"] = lr

    def before_run(self, runner):
        if self.scale_by_world_size:
            assert dist.is_available() and dist.is_initialized()
            lr_multiplier = dist.get_world_size()
        else:
            lr_multiplier = 1.0
        for group in _get_applied_param_groups(runner, self.apply_to):
            group["lr"] *= lr_multiplier
            group.setdefault("initial_lr", group["lr"])
        self.base_lr = [group["initial_lr"] for group in _get_applied_param_groups(runner, self.apply_to)]


class StepMultiOptimLrUpdateHook(_MultiOptimLrUpdateHook):

    def __init__(self, step, gamma, apply_to, scale_by_world_size=False,
                 by_epoch=True, warmup=None, warmup_iters=0, warmup_ratio=0.1):
        super(StepMultiOptimLrUpdateHook, self) \
            .__init__(apply_to, scale_by_world_size, by_epoch, warmup, warmup_iters, warmup_ratio)
        self.step = step
        self.gamma = gamma

    def get_lr(self, runner, base_lr):
        progress = runner.epoch if self.by_epoch else runner.iter

        if isinstance(self.step, int):
            return base_lr * (self.gamma ** (progress // self.step))

        exp = len(self.step)
        for i, s in enumerate(self.step):
            if progress < s:
                exp = i
                break
        return base_lr * self.gamma ** exp
