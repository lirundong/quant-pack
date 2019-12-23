# -*- coding: utf-8 -*-

from typing import List, Tuple, Optional
from mmcv.runner import Hook, Runner

VALID_QUANT_MODE = ("fp", "quant", "qw_fa", "fw_qa")
VALID_GRANULARITY = ("epoch", "iter")

IntervalT = List[Tuple[int, int]]

__all__ = ["EnableQuantAtIntervals", "IntervalWarmupedVariable", "ConstantVariable",
           "OptimAlterStep"]


def _in_intervals(i: int, intervals: IntervalT) -> Optional[Tuple[int, int]]:
    ret = None
    for (a, b) in intervals:
        if (b != -1 and a <= i < b) or a <= i:
            ret = (a, b)
    return ret


class EnableQuantAtIntervals(Hook):

    def __init__(self, quant_mode: str, granularity: str, intervals: IntervalT):
        assert quant_mode in VALID_QUANT_MODE
        assert granularity in VALID_GRANULARITY
        self.quant_mode = quant_mode
        self.granularity = granularity
        self.intervals = intervals

    def before_epoch(self, runner: Runner):
        if self.granularity == "epoch":
            if _in_intervals(runner.epoch, self.intervals):
                runner.model.quant_mode = (self.quant_mode, "fp")
            else:
                runner.model.quant_mode = ("fp", )

    def before_iter(self, runner: Runner):
        if self.granularity == "iter":
            if _in_intervals(runner.iter, self.intervals):
                runner.model.quant_mode = (self.quant_mode, "fp")
            else:
                runner.model.quant_mode = ("fp", )


class IntervalWarmupedVariable(Hook):

    def __init__(self, name, value, warmup_epochs, intervals):
        self.name = name
        self.target_value = value
        self.warmup_epochs = warmup_epochs
        self.intervals = intervals
        self.warmup_intervals = [(a, min(a + warmup_epochs, b)) for (a, b) in intervals]

    def before_run(self, runner: Runner):
        if not hasattr(runner, "named_vars"):
            runner.named_vars = dict()

    def before_iter(self, runner: Runner):
        if _in_intervals(runner.epoch, self.intervals):
            warmup_interval = _in_intervals(runner.epoch, self.warmup_intervals)
            if warmup_interval:
                iters_per_epoch = len(runner.data_loader)
                delta = self.target_value / self.warmup_epochs / iters_per_epoch
                gone_iters = (runner.epoch - warmup_interval[0]) * iters_per_epoch + runner.inner_iter
                runner.named_vars[self.name] = delta * gone_iters
            else:
                runner.named_vars[self.name] = self.target_value
        else:
            runner.named_vars[self.name] = 0


class ConstantVariable(Hook):

    def __init__(self, name, value):
        self.name = name
        self.value = value

    def before_run(self, runner: Runner):
        if not hasattr(runner, "named_vars"):
            runner.named_vars = dict()
        runner.named_vars[self.name] = self.value


class OptimAlterStep(Hook):

    def __init__(self, apply_to, alter_freq, intervals):
        self.apply_to = apply_to
        self.alter_freq = alter_freq
        self.intervals = intervals

    def after_iter(self, runner):
        loss = runner.outputs["loss"]
        runner.model.zero_grad()
        loss.backward()
        if _in_intervals(runner.epoch, self.intervals):
            optim_idx = int((runner.inner_iter // self.alter_freq) % len(self.apply_to))
            optim = runner.optimizer[self.apply_to[optim_idx]]
            optim.step()
        else:
            for name in self.apply_to:
                runner.optimizer[name].step()
