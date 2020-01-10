# -*- coding: utf-8 -*-

from .base_builder import HookBuilder


class ManualBiasCorrectionBuilder(HookBuilder):

    def __init__(self, hook_reg, target_layer, target_conf, relevant_bias):
        super(ManualBiasCorrectionBuilder, self).__init__(phases=("forward_pre", "forward"), hook_reg=hook_reg)
        assert target_conf in ("input", "weight")
        self.target_layer = target_layer
        self.target_conf = target_conf
        self.relevant_bias = relevant_bias

    def match(self, name, module):
        return name == self.target_layer

    def inject_at(self, quant_mode):
        return quant_mode != "fp"

    def _runtime_forward_pre_hook(self, module, input):
        if self.target_conf == "input":
            assert hasattr(module, "input_qconf")
            module.input_qconf._manual_bias = self.relevant_bias
            module.input_transform = module.input_qconf.transform
        elif self.target_conf == "weight":
            assert hasattr(module, "weight_qconf")
            module.weight_qconf._manual_bias = self.relevant_bias
            module.weight_transform = module.weight_qconf.transform

    def _runtime_forward_hook(self, module, input, output):
        if hasattr(module, "input_qconf"):
            module.input_qconf._manual_bias = None
        module.input_transform = module.input_qconf.transform
        if hasattr(module, "weight_qconf"):
            module.weight_qconf._manual_bias = None
            module.weight_transform = module.weight_qconf.transform
