# -*- coding: utf-8 -*-


class HookBuilder:

    def __init__(self, phases, hook_reg):
        if isinstance(phases, str):
            phases = (phases, )
        assert all(phase in ("forward", "forward_pre", "backward") for phase in phases)
        self._phases = phases
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

    def get_hooks(self):
        ret = []
        for phase in self._phases:
            if phase == "forward":
                ret.append(("register_forward_hook", self._runtime_forward_hook))
            elif phase == "forward_pre":
                ret.append(("register_forward_pre_hook", self._runtime_forward_pre_hook))
            elif phase == "backward":
                ret.append(("register_backward_hook", self._runtime_backward_hook))
        return ret
