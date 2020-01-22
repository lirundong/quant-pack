# -*- coding: utf-8 -*-

from collections import OrderedDict

from torch.nn.parallel import DistributedDataParallel
from mmcv.runner import Hook

from .activation_builder import SaveActivationBuilder, SaveAllValueBuilder
from .activation_post_process import CosineDistancePostProcess, RelativeErrorPostProcess
from .manual_bias_correction_builder import ManualBiasCorrectionBuilder

# TODO: refactor these registries to class decorators
BUILDERS = {
    SaveActivationBuilder.__name__: SaveActivationBuilder,
    SaveAllValueBuilder.__name__: SaveAllValueBuilder,
    ManualBiasCorrectionBuilder.__name__: ManualBiasCorrectionBuilder,
}

POST_PROCESS = {
    CosineDistancePostProcess.__name__: CosineDistancePostProcess,
    RelativeErrorPostProcess.__name__: RelativeErrorPostProcess,
}


class RuntimeHook(Hook):

    def __init__(self, intervals, hook_builders):
        self.intervals = intervals
        self.hook_regs = OrderedDict()
        self.enable_reg = OrderedDict()
        self.named_builders = OrderedDict()
        self.named_handles = OrderedDict()
        self.is_ddp = False
        self.enabled_at_this_iter = False

        for builder_cfg in hook_builders:
            hook_name = builder_cfg["name"]
            hook_cls = builder_cfg["type"]
            hook_args = builder_cfg["args"]
            need_reg = hook_args.pop("need_reg", False)
            if need_reg:
                hook_reg = OrderedDict()
                self.hook_regs[hook_name] = hook_reg
            else:
                hook_reg = None
            if hook_cls not in BUILDERS:
                hook_cls += "Builder"
            hook_builder = BUILDERS[hook_cls](hook_reg=hook_reg, enable_reg=self.enable_reg, **hook_args)
            self.named_builders[hook_name] = hook_builder
            self.enable_reg[id(hook_builder)] = False

    def before_run(self, runner):
        module = runner.model.module
        if isinstance(module, DistributedDataParallel):
            self.is_ddp = True
        else:
            self.is_ddp = False
            for n, m in module.named_modules():
                for builder_name, builder in self.named_builders.items():
                    if builder.match(n, m):
                        for register_method, hook in builder.get_hooks():
                            handle = getattr(m, register_method)(hook)
                            self.named_handles[f"{builder_name}_{register_method}"] = handle

    def before_iter(self, runner):
        if self.every_n_inner_iters(runner, self.intervals):
            self.enabled_at_this_iter = True
        else:
            self.enabled_at_this_iter = False
            for k in self.enable_reg.keys():
                self.enable_reg[k] = False

    def update_hooks(self, quant_mode):
        activated_builders = []
        if self.enabled_at_this_iter:
            for name, builder in self.named_builders.items():
                activated = builder.inject_at(quant_mode)
                self.enable_reg[name] = activated
                if activated:
                    activated_builders.append(builder)
        if self.is_ddp:
            return activated_builders


class WithPostprocessRuntimeHook(RuntimeHook):

    def __init__(self, intervals, hook_builders, post_process):
        super(WithPostprocessRuntimeHook, self).__init__(intervals, hook_builders)
        self.post_process = OrderedDict()
        for process_cfg in post_process:
            process_name = process_cfg["name"]
            process_cls = process_cfg["type"]
            process_args = process_cfg["args"]
            if process_cls not in POST_PROCESS:
                process_cls += "PostProcess"
            process = POST_PROCESS[process_cls](**process_args)
            self.post_process[process_name] = process

    def after_val_iter(self, runner):
        if self.enabled_at_this_iter:
            plot_buffer = OrderedDict()
            for name, process in self.post_process.items():
                result = process.after_iter(self.hook_regs, runner.outputs)
                plot_buffer[name] = result
            for _, reg in self.hook_regs.items():
                reg.clear()
            runner.log_buffer.output["plot"] = plot_buffer
