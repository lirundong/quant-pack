# -*- coding: utf-8 -*-

from collections import defaultdict
from typing import Dict, Callable, Union, List

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

__all__ = ["VisDiagnoser"]

ModuleT = nn.Module
TensorT = torch.Tensor

PreFwdHookT = Callable[[ModuleT, TensorT], None]
FwdHookT = Callable[[ModuleT, TensorT, TensorT], None]
BkwdHookT = Callable[[ModuleT, TensorT, TensorT], Union[None, TensorT]]
HookT = Union[PreFwdHookT, FwdHookT, BkwdHookT]


# TODO: base class for Diagnoser
class VisDiagnoser:

    def __init__(self,
                 module: ModuleT,
                 logger: SummaryWriter,
                 tasks,
                 diag_layers: Union[List[str], None] = None):
        # TODO(Rundong):
        #   - [ ] support multiple input modules for comparision;
        #   - [x] make sure module-name mapping and enable flags are visible to diagnose hooks;
        #   - [ ] implement interactive diagnose methods;
        #   - [ ] correct signature for registry arguments

        self.module = module
        self.tasks = tasks
        self.logger = logger
        self.diag_layers = diag_layers

        self.pre_fwd_hook_enabled = False
        self.fwd_hook_enabled = False
        self.bkwd_hook_enabled = False
        self.diag_all_layers = self.diag_layers is None

        self.handles = []
        self.step_done_calls = []
        self.call_counter = 0

        self.register_hooks()

    def register_hooks(self):

        def _do_register(module, builder):
            if stage_name == "pre_fwd":
                func = builder.get_pre_fwd_hook()
                h = module.register_forward_pre_hook(func)
            elif stage_name == "fwd":
                func = builder.get_fwd_hook()
                h = module.register_forward_hook(func)
            elif stage_name == "bkwd":
                func = builder.get_bkwd_hook()
                h = module.register_backward_hook(func)
            else:
                raise ValueError(f"unknown stage name: {stage_name}")

            self.handles.append(h)
            self.step_done_calls.append(builder.get_step_done())

        for stage_name, stage_tasks in self.tasks.items():
            assert stage_name in ("pre_fwd", "fwd", "bkwd")
            for task_name, task_builder in stage_tasks.items():
                builder = task_builder(self, self.logger)
                for n, m in self.module.named_modules():
                    if self.diag_all_layers or n in self.diag_layers:
                        _do_register(m, builder)

    def remove_hooks(self):
        if len(self.handles) > 0:
            for h in self.handles:
                h.remove()

    def forward(self, inputs, step):
        ret = self.module(inputs)

        if len(self.step_done_calls) > 0:
            for step_done in self.step_done_calls:
                step_done(step)

        return ret

    def __call__(self, inputs, step=None):
        self.call_counter += 1
        if step is None:
            step = self.call_counter

        return self.forward(inputs, step)

    def __getattr__(self, item):
        return getattr(self.module, item)
