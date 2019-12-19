# -*- coding: utf-8 -*-

from mmcv.runner import Hook


class RuntimeInjectableHook(Hook):

    def __init__(self, phase):
        assert phase in ("forward", "forward_pre", "backward")
        self._phase = phase

    @staticmethod
    def _runtime_forward_hook(module, input, output):
        raise NotImplementedError()

    @staticmethod
    def _runtime_forward_pre_hook(module, input):
        raise NotImplementedError()

    @staticmethod
    def _runtime_backward_hook(module, grad_input, grad_output):
        raise NotImplementedError()

    def match(self, name, module):
        raise NotImplementedError()

    def get_hook(self):
        if self._phase is "forward":
            return "register_forward_hook", self._runtime_forward_hook
        elif self._phase is "forward_pre":
            return "register_forward_pre_hook", self._runtime_forward_pre_hook
        elif self._phase is "backward":
            return "register_backward_hook", self._runtime_backward_hook
        raise RuntimeError(f"invalid hooking phase: {self._phase}")
