# -*- coding: utf-8 -*-

import torch
import torch.distributed as dist

from quant_pack.core.quant.config import QuantMode
from .base_builder import HookBuilder


class ActivationCalibrationBuilder(HookBuilder):

    def __init__(self, hook_reg, enable_reg, percentile=0.99):
        super(ActivationCalibrationBuilder, self).__init__("forward", hook_reg, enable_reg)
        self.percentile = percentile

    def match(self, name, module):
        return hasattr(module, "input_qconf")

    def inject_at(self, quant_mode):
        return QuantMode.Calib in quant_mode

    def _runtime_forward_hook(self, module, input, output):
        if isinstance(input, tuple):
            input = input[0]
        input, _ = input.reshape(-1).sort()
        n = input.numel()
        k = int(n * self.percentile)
        if dist.is_available() and dist.is_initialized():
            bounds = torch.tensor([input[k], input[n - k]], device=input.device).div_(dist.get_world_size())
            dist.all_reduce(bounds, dist.ReduceOp.SUM)
            ub, lb = bounds
        else:
            ub, lb = input[k], input[n - k]
        assert lb < ub
        module.a_lb.copy_(lb)
        module.a_ub.copy_(ub)
