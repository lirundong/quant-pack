# -*- coding: utf-8 -*-

from math import sqrt

from ._registry import register

__all__ = ["AccGradNorm"]


class Task:
    """Observer class, hold internal states for each diagnose task."""

    def __init__(self, diagnoser, task_name, stage):
        assert stage in ("pre_fwd", "fwd", "bkwd", "tensor")
        self.diagnoser = diagnoser
        self.task_name = task_name
        self.stage = stage

        self.is_enabled = False  # this flag should only be modified from `diagnoser`
        self.step_done_required = False
        self.diagnoser.register_task(self)

    def __del__(self):
        try:
            self.diagnoser.unregister_task(self)
        except ValueError as e:
            pass

    def is_enabled_at(self, step):
        raise NotImplementedError

    def step_done(self, *args, **kwargs):
        raise NotImplementedError

    def get_tensor_hook(self):
        raise NotImplementedError

    def get_pre_fwd_hook(self):
        raise NotImplementedError

    def get_fwd_hook(self):
        raise NotImplementedError

    def get_bkwd_hook(self):
        raise NotImplementedError


@register("grad_norm")
class AccGradNorm(Task):

    def __init__(self, diagnoser, task_name, stage, frequency):
        super(AccGradNorm, self).__init__(diagnoser, task_name, stage)
        self.frequency = int(frequency)
        self.step_done_required = True
        self.grad_norm = 0.
        self.eps = 1e-6

    def is_enabled_at(self, step):
        if self.frequency <= 0:
            return False
        else:
            return int(step) % self.frequency == 0

    def step_done(self, step, logger):
        if self.is_enabled:
            g_norm = sqrt(self.grad_norm + self.eps)
            logger.add_scalar(self.task_name, g_norm, step)
            self.grad_norm = 0.

    def get_tensor_hook(self):

        def _grad_norm_hook(grad):
            if self.is_enabled:
                self.grad_norm += grad.data.norm().item() ** 2

        return _grad_norm_hook

