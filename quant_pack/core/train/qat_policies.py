# -*- coding: utf-8 -*-

from collections import OrderedDict

import torch
from torch.nn.parallel import DistributedDataParallel

from typing import List, Tuple, Optional
from mmcv.runner import Hook, Runner

from quant_pack.core.wrapper.hook.activation_builder import HijackModuleOutputBuilder
from quant_pack.core.quant.config import QuantMode

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
                 always_enable_fp: bool = False, calibrate_steps: int = 1, calibrate_cfg: dict = {}):
        assert quant_mode in VALID_QUANT_MODE
        assert granularity in VALID_GRANULARITY
        if always_enable_fp and quant_mode != "fp":
            quant_mode = (quant_mode, "fp")
        else:
            quant_mode = (quant_mode,)
        self.quant_mode = tuple(QuantMode.get(m) for m in quant_mode)
        self.granularity = granularity
        self.intervals = intervals
        self.calibrate_steps = calibrate_steps
        self.calibrate_cfg = calibrate_cfg
        self.do_calibration_at = min(i[0] for i in intervals)

    def _switch_quant_mode(self, runner):
        if _in_intervals(runner.epoch, self.intervals):
            quant_mode = self.quant_mode
            in_qat = True
        else:
            quant_mode = (QuantMode.FWFA,)
            in_qat = False
        if isinstance(runner.model.module, DistributedDataParallel):
            find_unused_param = QuantMode.QWQA not in quant_mode
            if not find_unused_param:
                for m in runner.model._quant_submodules:
                    if m.weight_qconf.retain_fp or m.input_qconf.retain_fp:
                        find_unused_param = True
                        break
            runner.model.module.find_unused_parameters = find_unused_param
        if runner.model.quant_mode != quant_mode:
            runner.logger.info(f"switch quant mode to: {quant_mode}, in QAT: {in_qat}")
        runner.model.quant_mode = quant_mode
        runner.model._in_qat = in_qat

    def _do_calibration(self, runner):
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        runner.model.do_calibration(runner, self.calibrate_steps, self.calibrate_cfg, device, runner.runtime_hook)

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

    def __init__(self, quant_mode, calibrate_steps=1, calibrate_cfg=None):
        if isinstance(quant_mode, str):
            quant_mode = (quant_mode, )
        self.quant_mode = tuple(QuantMode.get(m) for m in quant_mode)
        self.calibrate_steps = calibrate_steps
        self.calibrate_cfg = {} if calibrate_cfg is None else calibrate_cfg

    def _do_calibration(self, runner):
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
        runner.model.do_calibration(runner, self.calibrate_steps, self.calibrate_cfg, device, runner.runtime_hook)

    def before_run(self, runner):
        runner.model.quant_mode = self.quant_mode
        if isinstance(runner.model.module, DistributedDataParallel):
            runner.model.module.find_unused_parameters = QuantMode.QWQA not in self.quant_mode

    def before_train_epoch(self, runner):
        runner.model._in_qat = any(QuantMode.FWFA not in mode for mode in self.quant_mode)
        if runner.epoch == 0 and runner.model._in_qat:
            self._do_calibration(runner)


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

    def __init__(self, apply_to, alter_freq, intervals, loss_seq=None):
        self.apply_to = apply_to
        self.alter_freq = alter_freq
        self.intervals = intervals
        self.loss_seq = loss_seq

    @staticmethod
    def check_and_backward(loss, retain_graph):
        if torch.is_tensor(loss):
            loss.backward(retain_graph=retain_graph)

    def after_train_iter(self, runner):
        if _in_intervals(runner.epoch, self.intervals):
            if self.alter_freq > 0:
                optim_idx = int((runner.inner_iter // self.alter_freq) % len(self.apply_to))
                optims = [runner.optimizer[self.apply_to[optim_idx]], ]
            else:
                optims = [runner.optimizer[n] for n in self.apply_to]
        else:
            optims = [runner.optimizer[self.apply_to[0]], ]

        for optim in optims:
            optim.zero_grad()
        if self.loss_seq is not None:
            for loss_name in self.loss_seq:
                retain_graph = loss_name is not self.loss_seq[-1]
                self.check_and_backward(runner.outputs[loss_name], retain_graph)
        else:
            loss = sum(v for k, v in runner.outputs.items() if k.endswith("_loss"))
            loss.backward()
        for optim in optims:
            optim.step()


class HijackModuleOutput(Hook):

    def __init__(self, module_name, output_name, detach_fp=True):
        self.module_name = module_name
        self.hook_name = f"hijack_{self.module_name}"
        self.output_name = output_name
        self.detach_fp = detach_fp

    def before_run(self, runner):
        if not hasattr(runner.model, "runtime_hooks"):
            runner.model.runtime_hooks = OrderedDict()

    def before_train_iter(self, runner):
        # priority should be "LOW" to make sure `_in_qat` is ready
        if runner.model._in_qat:
            runner.model.runtime_hooks[self.hook_name] = HijackModuleOutputBuilder(self.module_name, self.output_name)

    def after_train_iter(self, runner):
        if self.hook_name in runner.model.runtime_hooks:
            runner.model.runtime_hooks.pop(self.hook_name)
