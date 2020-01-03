# -*- coding: utf-8 -*-

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from typing import List, Tuple, Optional
from mmcv.runner import Hook, Runner

VALID_QUANT_MODE = ("fp", "quant", "qw_fa", "fw_qa")
VALID_GRANULARITY = ("epoch", "iter")

IntervalT = List[Tuple[int, int]]


def _in_intervals(i: int, intervals: IntervalT) -> Optional[Tuple[int, int]]:
    ret = None
    for (a, b) in intervals:
        if (b != -1 and a <= i < b) or a <= i:
            ret = (a, b)
    return ret


class EnableQuantAtIntervals(Hook):

    def __init__(self, quant_mode: str, granularity: str, intervals: IntervalT,
                 always_enable_fp: bool = False, calibrate_steps: int = 1, calibrate_percentile: float = 0.99):
        assert quant_mode in VALID_QUANT_MODE
        assert granularity in VALID_GRANULARITY
        if always_enable_fp and quant_mode != "fp":
            quant_mode = (quant_mode, "fp")
        else:
            quant_mode = (quant_mode,)
        self.quant_mode = quant_mode
        self.granularity = granularity
        self.intervals = intervals
        self.calibrate_steps = calibrate_steps
        self.calibrate_percentile = calibrate_percentile
        self.do_calibration_at = min(i[0] for i in intervals)

    def _switch_quant_mode(self, runner):
        if _in_intervals(runner.epoch, self.intervals):
            quant_mode = self.quant_mode
        else:
            quant_mode = ("fp",)
        if runner.model.quant_mode != quant_mode and isinstance(runner.model.module, DistributedDataParallel):
            runner.model.module.find_unused_parameters = "quant" not in quant_mode
        runner.model.quant_mode = quant_mode

    def _do_calibration(self, runner):
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        runner.model.do_calibration(runner, self.calibrate_steps, self.calibrate_percentile, device)

    def before_train_epoch(self, runner: Runner):
        if self.granularity == "epoch":
            if runner.epoch == self.do_calibration_at:
                self._do_calibration(runner)
            self._switch_quant_mode(runner)

    def before_train_iter(self, runner: Runner):
        if self.granularity == "iter":
            if runner.iter == self.do_calibration_at:
                self._do_calibration(runner)
            self._switch_quant_mode(runner)


class SetupQuantOnce(Hook):

    def __init__(self, quant_mode):
        if isinstance(quant_mode, str):
            quant_mode = (quant_mode, )
        self.quant_mode = quant_mode

    def before_run(self, runner):
        runner.model.quant_mode = self.quant_mode
        runner.model.module.find_unused_parameters = "quant" not in self.quant_mode


class IntervalWarmupedVariable(Hook):

    def __init__(self, name, value, warmup_iters, intervals):
        self.name = name
        self.target_value = value
        self.warmup_iters = warmup_iters
        self.intervals = intervals

    def before_run(self, runner: Runner):
        if not hasattr(runner, "named_vars"):
            runner.named_vars = dict()

    def before_train_epoch(self, runner):
        current_interval = _in_intervals(runner.epoch, self.intervals)
        if current_interval:
            iters_per_epoch = len(runner.data_loader)
            warmup_start_iter = current_interval[0] * iters_per_epoch
            warmup_done_iter = warmup_start_iter + self.warmup_iters
        else:
            warmup_start_iter = warmup_done_iter = None
        self.warmup_start_iter = warmup_start_iter
        self.warmup_done_iter = warmup_done_iter
        self.value_enabled = current_interval is not None
        self.warmup_delta = self.target_value / self.warmup_iters

    def before_train_iter(self, runner: Runner):
        if self.value_enabled:
            if runner.iter < self.warmup_done_iter:
                gone_iters = runner.iter - self.warmup_start_iter
                runner.named_vars[self.name] = self.warmup_delta * gone_iters
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

    def after_train_iter(self, runner):
        if _in_intervals(runner.epoch, self.intervals):
            optim_idx = int((runner.inner_iter // self.alter_freq) % len(self.apply_to))
            optims = [runner.optimizer[self.apply_to[optim_idx]], ]
        else:
            optims = [runner.optimizer[name] for name in self.apply_to]

        for optim in optims:
            optim.zero_grad()
        loss = runner.outputs["loss"]
        loss.backward()
        for optim in optims:
            optim.step()
