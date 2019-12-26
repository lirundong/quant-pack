# -*- coding: utf-8 -*-

import re

import torch.nn as nn

from .base_builder import HookBuilder


class SaveActivationBuilder(HookBuilder):

    def __init__(self, hook_reg, target_cls, inject_at_mode):
        super(SaveActivationBuilder, self).__init__(phase="forward", hook_reg=hook_reg)
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

    def _runtime_forward_hook(self, module, input, output):
        output = output.detach().clone()
        name = self.name_reg[id(module)]
        self._reg[name] = output


class SaveAllValueBuilder(SaveActivationBuilder):

    def _runtime_forward_hook(self, module, input, output):
        if isinstance(input, tuple):
            assert len(input) == 1
            input = input[0]
        input = input.detach().clone()
        output = output.detach().clone()
        weight = module.weight.detach().clone()
        name = self.name_reg[id(module)]
        self._reg[name] = {
            "input": input,
            "output": output,
            "weight": weight,
            "type": module.__class__.__name__,
        }
        if isinstance(module, nn.Conv2d):
            self._reg[name]["param"] = {
                "kernel_size": module.kernel_size,
                "dilation": module.dilation,
                "padding": module.padding,
                "stride": module.stride,
            }
