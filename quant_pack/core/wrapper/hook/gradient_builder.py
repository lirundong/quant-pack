# -*- coding: utf-8 -*-

import re
from collections import OrderedDict

from .base_builder import HookBuilder


class HijackGradientBuilder(HookBuilder):

    def __init__(self, hook_reg, enable_reg, loss_seq, layers):
        super(HijackGradientBuilder, self).__init__("backward", hook_reg, enable_reg)
        self.loss_seq = loss_seq
        self.layers = re.compile(layers)
        self.name_reg = {}
        self.count_reg = {}

    def match(self, name, module):
        match = self.layers.match(name)
        if match:
            self.name_reg[id(module)] = name
            self.count_reg[id(module)] = 0
        return match

    def inject_at(self, quant_mode):
        # enable this hook once `interval` met
        return True

    def _runtime_backward_hook(self, module, grad_input, grad_output):
        loss_name = self.loss_seq[self.count_reg[id(module)]]
        module_name = self.name_reg[id(module)]
        grad_output = grad_output[0].detach().clone()
        if module_name not in self._reg:
            self._reg[module_name] = OrderedDict({loss_name: grad_output})
        else:
            self._reg[module_name][loss_name] = grad_output
        self.count_reg[id(module)] += 1
        self.count_reg[id(module)] %= len(self.loss_seq)


class HijackTensorGradientBuilder(HookBuilder):

    def __init__(self, hook_reg, enable_reg, loss_seq, tensor_name):
        super(HijackTensorGradientBuilder, self).__init__("tensor", hook_reg, enable_reg)
        self.loss_seq = loss_seq
        self.tensor_name = re.compile(tensor_name)
        self.loss_count = 0

    def match(self, name, param):
        if isinstance(self.tensor_name, str):
            return False  # already have match
        match = self.tensor_name.match(name)
        if match:
            self.tensor_name = name
        return match

    def _runtime_tensor_hook(self, tensor_grad):
        loss_name = self.loss_seq[self.loss_count]
        tensor_grad = tensor_grad.detach().clone()
        if self.tensor_name not in self._reg:
            self._reg[self.tensor_name] = OrderedDict({loss_name: tensor_grad})
        else:
            self._reg[self.tensor_name][loss_name] = tensor_grad
        self.loss_count += 1
        self.loss_count %= len(self.loss_seq)
