# -*- coding: utf-8 -*-


class HookBuilder:

    def __init__(self, phases, hook_reg, enable_reg):
        if isinstance(phases, str):
            phases = (phases, )
        assert all(phase in ("forward", "forward_pre", "backward", "tensor") for phase in phases)
        self._phases = phases
        self._reg = hook_reg
        self._enable_reg = enable_reg

    def _runtime_forward_hook(self, module, input, output):
        raise NotImplementedError()

    def _runtime_forward_pre_hook(self, module, input):
        raise NotImplementedError()

    def _runtime_backward_hook(self, module, grad_input, grad_output):
        raise NotImplementedError()

    def _runtime_tensor_hook(self, tensor_grad):
        raise NotImplementedError()

    @property
    def enabled(self):
        if self._enable_reg is None:
            return True
        else:
            return self._enable_reg[id(self)]

    def forward_hook(self, module, input, output):
        if self.enabled:
            return self._runtime_forward_hook(module, input, output)

    def forward_pre_hook(self, module, input):
        if self.enabled:
            return self._runtime_forward_pre_hook(module, input)

    def backward_hook(self, module, grad_input, grad_output):
        if self.enabled:
            return self._runtime_backward_hook(module, grad_input, grad_output)

    def tensor_hook(self, tensor_grad):
        if self.enabled:
            return self._runtime_tensor_hook(tensor_grad)

    def match(self, *args, **kwargs):
        raise NotImplementedError()

    def inject_at(self, *args, **kwargs):
        raise NotImplementedError()

    def get_hooks(self):
        ret = []
        for phase in self._phases:
            if phase == "forward":
                ret.append(("register_forward_hook", self.forward_hook))
            elif phase == "forward_pre":
                ret.append(("register_forward_pre_hook", self.forward_pre_hook))
            elif phase == "backward":
                ret.append(("register_backward_hook", self.backward_hook))
            elif phase == "tensor":
                ret.append(("register_hook", self.backward_hook))
        return ret
