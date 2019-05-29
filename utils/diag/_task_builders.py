# -*- coding: utf-8 -*-

from math import sqrt
from typing import Callable

from tensorboardX import SummaryWriter

from .registry import register


class TaskBuilderBase:

    def __init__(self,
                 is_enabled: Callable,
                 tb_logger: SummaryWriter):
        self.is_enabled = is_enabled
        self.tb_logger = tb_logger

    def get_pre_fwd_hook(self, *args, **kwargs):
        raise NotImplementedError

    def get_fwd_hook(self, *args, **kwargs):
        raise NotImplementedError

    def get_bkwd_hook(self, *args, **kwargs):
        raise NotImplementedError

    def get_step_done(self, *args, **kwargs):
        raise NotImplementedError


@register("grad_norm")
class AccGradNorm(TaskBuilderBase):

    def __init__(self,
                 is_enabled: Callable,
                 tb_logger: SummaryWriter,
                 prefix: str):
        super(AccGradNorm, self).__init__(is_enabled, tb_logger)

        self.global_norm = 0
        self.eps = 1e-6
        self.prefix = prefix

    def get_bkwd_hook(self):

        def _grad_norm_hook(grad):
            if self.is_enabled():
                self.global_norm += grad.data.norm().item() ** 2

        return _grad_norm_hook

    def get_step_done(self):

        def _step_done(step):
            if self.is_enabled():
                g_norm = sqrt(self.global_norm + self.eps)
                self.tb_logger.add_scalar(self.prefix, g_norm, step)
                self.global_norm = 0

        return _step_done
