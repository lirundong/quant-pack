# -*- coding: utf-8 -*-

import re

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
