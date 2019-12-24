# -*- coding: utf-8 -*-


class HookBuilder:

    def __init__(self, phase, hook_reg):
        assert phase in ("forward", "forward_pre", "backward")
        self._phase = phase
        self._reg = hook_reg

    def _runtime_forward_hook(self, module, input, output):
        raise NotImplementedError()

    def _runtime_forward_pre_hook(self, module, input):
        raise NotImplementedError()

    def _runtime_backward_hook(self, module, grad_input, grad_output):
        raise NotImplementedError()

    def match(self, name, module):
        raise NotImplementedError()

    def inject_at(self, *args, **kwargs):
        raise NotImplementedError()

    def get_hook(self):
        if self._phase is "forward":
            return "register_forward_hook", self._runtime_forward_hook
        elif self._phase is "forward_pre":
            return "register_forward_pre_hook", self._runtime_forward_pre_hook
        elif self._phase is "backward":
            return "register_backward_hook", self._runtime_backward_hook
        raise RuntimeError(f"invalid hooking phase: {self._phase}")
