# -*- coding: utf-8 -*-

import re
from collections import OrderedDict

from .base_builder import HookBuilder
from quant_pack.core.quant.config import QuantMode


def _to_cpu(tensor):
    return tensor.detach().cpu().clone()


class CollectLayerParamBuilder(HookBuilder):

    def __init__(self, hook_reg, enable_reg, layers, inject_at_mode):
        """

        Args:
            hook_reg:
            enable_reg:
            layers (dict): {"regexp_of_layer_name": [list of param name]}
            inject_at_mode:
        """
        super(CollectLayerParamBuilder, self).__init__("forward", hook_reg, enable_reg)
        if not isinstance(inject_at_mode, QuantMode):
            inject_at_mode = QuantMode.get(inject_at_mode)
        self.target_layers = [re.compile(n) for n in layers.keys()]
        self.param_names = {re.compile(k): v for k, v in layers.items()}
        self.inject_at_mode = inject_at_mode
        self.name_reg = {}

    def match(self, name, module):
        match = False
        for r, param_names in self.param_names.items():
            if r.match(name):
                match = True
                self.name_reg[id(module)] = [f"{name}/{p}" for p in param_names]
                break
        return match

    def inject_at(self, quant_mode):
        return self.inject_at_mode in quant_mode

    def _runtime_forward_hook(self, module, input, output):
        for param_names in self.name_reg[id(module)]:
            layer_name, p_name = param_names.split("/")
            param = _to_cpu(getattr(module, p_name))
            if layer_name not in self._reg:
                self._reg[layer_name] = OrderedDict({p_name: param})
            else:
                self._reg[layer_name][p_name] = param
