# -*- coding: utf-8 -*-

import re
from collections import OrderedDict

import torch
import torch.nn as nn

from .base_builder import HookBuilder


def copy_to_cpu(x):
    cpu = torch.device("cpu")
    return x.detach().to(cpu).clone()


class SaveActivationBuilder(HookBuilder):

    def __init__(self, hook_reg, target_cls, inject_at_mode):
        super(SaveActivationBuilder, self).__init__(phases=("forward_pre", "forward"), hook_reg=hook_reg)
        self.target_cls = re.compile(target_cls)
        self.inject_at_mode = inject_at_mode
        self.name_reg = {}

    def match(self, name, module):
        if self.target_cls.match(module.__class__.__name__):
            self.name_reg[id(module)] = name
            return True
        else:
            return False

    def inject_at(self, quant_mode):
        return quant_mode == self.inject_at_mode

    def _runtime_forward_pre_hook(self, module, input):
        module.gather_data = True
        module.gather_buffer = OrderedDict()

    def _runtime_forward_hook(self, module, input, output):
        output = copy_to_cpu(output)
        name = self.name_reg[id(module)]
        self._reg[name] = output


class SaveAllValueBuilder(SaveActivationBuilder):

    def _runtime_forward_hook(self, module, input, output):
        name = self.name_reg[id(module)]
        self._reg[name] = {
            "type": module.__class__.__name__,
            "input_qconf": module.input_qconf.params,
            "weight_qconf": module.weight_qconf.params,
        }
        if isinstance(module, nn.Conv2d):
            self._reg[name]["param"] = {
                "kernel_size": module.kernel_size,
                "dilation": module.dilation,
                "padding": module.padding,
                "stride": module.stride,
            }
        for k, v in module.gather_buffer.items():
            v = copy_to_cpu(v)
            self._reg[name][k] = v
        module.gather_buffer.clear()
        module.gather_data = False
