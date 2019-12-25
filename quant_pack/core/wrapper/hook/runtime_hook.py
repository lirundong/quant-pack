# -*- coding: utf-8 -*-

from collections import OrderedDict

from mmcv.runner import Hook

from .activation_builder import SaveActivationBuilder
from .activation_post_process import CosineDistancePostProcess

# TODO: refactor these registries to class decorators
BUILDERS = {
    SaveActivationBuilder.__name__: SaveActivationBuilder,
}

POST_PROCESS = {
    CosineDistancePostProcess.__name__: CosineDistancePostProcess,
}


class InjectRuntimeHook(Hook):

    def __init__(self, intervals, hook_builders, post_process):
        self.intervals = intervals
        self.hook_reg = OrderedDict()
        self.named_hooks = OrderedDict()
        self.post_process = OrderedDict()

        for builder_cfg in hook_builders:
            hook_name = builder_cfg["name"]
            hook_cls = builder_cfg["type"]
            hook_args = builder_cfg["args"]
            hook_reg = OrderedDict()
            if hook_cls not in BUILDERS:
                hook_cls += "Builder"
            hook = BUILDERS[hook_cls](hook_reg=hook_reg, **hook_args)
            self.hook_reg[hook_name] = hook_reg
            self.named_hooks[hook_name] = hook

        for process_cfg in post_process:
            process_name = process_cfg["name"]
            process_cls = process_cfg["type"]
            process_args = process_cfg["args"]
            if process_cls not in POST_PROCESS:
                process_cls += "PostProcess"
            process = POST_PROCESS[process_cls](**process_args)
            self.post_process[process_name] = process

    def before_run(self, runner):
        runner.model.runtime_hooks = OrderedDict()

    def before_iter(self, runner):
        if self.every_n_iters(runner, self.intervals) or self.end_of_epoch(runner):
            runner.model.runtime_hooks.update(self.named_hooks)
            runner.log_buffer.output["plot"] = OrderedDict()

    def after_iter(self, runner):
        if self.every_n_iters(runner, self.intervals) or self.end_of_epoch(runner):
            plot_buffer = runner.log_buffer.output["plot"]
            for name, process in self.post_process.items():
                result = process.after_iter(self.hook_reg)
                plot_buffer[name] = result
            for _, reg in self.hook_reg.items():
                reg.clear()
            runner.model.runtime_hooks.clear()
