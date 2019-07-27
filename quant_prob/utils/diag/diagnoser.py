# -*- coding: utf-8 -*-

from collections import defaultdict
from typing import Dict, Callable, Union, List

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

__all__ = ["VisDiagnoser"]

ModuleT = nn.Module
TensorT = torch.Tensor

PreFwdHookT = Callable[[ModuleT, TensorT], None]
FwdHookT = Callable[[ModuleT, TensorT, TensorT], None]
BkwdHookT = Callable[[ModuleT, TensorT, TensorT], Union[None, TensorT]]
HookT = Union[PreFwdHookT, FwdHookT, BkwdHookT]


class Diagnoser:
    """Observable diagnose registry.

    TODO:
        - (priority) check whether `module` is DDP
        - refactor this base-class into interface;
    """

    def __init__(self, module):
        self.module = module

        self.tasks = []
        self.handles = []
        self.call_counter = 0
        self.enabled_layers = None

    def register_task(self, task):
        self.tasks.append(task)

    def unregister_task(self, task):
        self.tasks.remove(task)

    def register_hooks(self):
        # TODO: check `enabled_layers`
        stage_hooks = defaultdict(list)

        for task in self.tasks:
            stage = task.stage
            if stage == "pre_fwd":
                hook = task.get_pre_fwd_hook()
            elif stage == "fwd":
                hook = task.get_fwd_hook()
            elif stage == "bkwd":
                hook = task.get_bkwd_hook()
            else:
                hook = task.get_tensor_hook()
            stage_hooks[stage].append(hook)

        for n, m in self.module.named_modules():
            for hook in stage_hooks["pre_fwd"]:
                h = m.register_forward_pre_hook(hook)
                self.handles.append(h)
            for hook in stage_hooks["fwd"]:
                h = m.register_forward_hook(hook)
                self.handles.append(h)
            for hook in stage_hooks["bkwd"]:
                h = m.register_backward_hook(hook)
                self.handles.append(h)

        for n, p in self.module.named_parameters():
            for hook in stage_hooks["tensor"]:
                h = p.register_hook(hook)
                self.handles.append(h)

    def remove_hooks(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def update_task_state(self, step=None):
        if step is None:
            step = self.call_counter

        for task in self.tasks:
            is_enabled = task.is_enabled_at(step)
            task.is_enabled = is_enabled

    def step_done(self, step=None):
        if step is None:
            step = self.call_counter

        for task in self.tasks:
            if task.step_done_required:
                task.step_done(step)


class VisDiagnoser(nn.Module, Diagnoser):

    def __init__(self,
                 module: ModuleT,
                 logger: SummaryWriter,
                 enabled_layers: Union[List[str], None] = None):
        # TODO(Rundong):
        #   - [ ] support multiple input modules for comparision;
        #   - [x] make sure module-name mapping and enable flags are visible to diagnose hooks;
        #   - [ ] implement interactive diagnose methods;
        #   - [ ] correct signature for registry arguments

        nn.Module.__init__(self)
        Diagnoser.__init__(self, module)

        self.logger = logger
        self.enabled_layers = enabled_layers

    def forward(self, *args, **kwargs):
        self.call_counter += 1
        return self.module(*args, **kwargs)

    def step_done(self, step=None):
        if step is None:
            step = self.call_counter

        for task in self.tasks:
            if task.step_done_required and self.logger is not None:
                task.step_done(step, self.logger)

    def __getattr__(self, item):
        try:
            return super(VisDiagnoser, self).__getattr__(item)
        except AttributeError:
            return getattr(self.module, item)
