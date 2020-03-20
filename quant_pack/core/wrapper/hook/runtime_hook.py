# -*- coding: utf-8 -*-

from collections import OrderedDict

from torch.nn.parallel import DistributedDataParallel
from mmcv.runner import Hook

from .activation_builder import SaveActivationBuilder, SaveAllValueBuilder
from .manual_bias_correction_builder import ManualBiasCorrectionBuilder
from .calibration_builder import ActivationCalibrationBuilder
from .gradient_builder import HijackGradientBuilder, HijackTensorGradientBuilder
from .param_builder import CollectLayerParamBuilder
from .activation_post_process import CosineDistancePostProcess, RelativeErrorPostProcess
from .gradient_post_process import MultiLossGradDist
from .param_post_process import ParamPassThrough

# TODO: refactor these registries to class decorators
BUILDERS = {
    SaveActivationBuilder.__name__: SaveActivationBuilder,
    SaveAllValueBuilder.__name__: SaveAllValueBuilder,
    ManualBiasCorrectionBuilder.__name__: ManualBiasCorrectionBuilder,
    ActivationCalibrationBuilder.__name__: ActivationCalibrationBuilder,
    HijackGradientBuilder.__name__: HijackGradientBuilder,
    HijackTensorGradientBuilder.__name__: HijackTensorGradientBuilder,
    CollectLayerParamBuilder.__name__: CollectLayerParamBuilder,
}

POST_PROCESS = {
    CosineDistancePostProcess.__name__: CosineDistancePostProcess,
    RelativeErrorPostProcess.__name__: RelativeErrorPostProcess,
    MultiLossGradDist.__name__: MultiLossGradDist,
    ParamPassThrough.__name__: ParamPassThrough,
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
            self.add_builder(builder_cfg)

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
                            self.named_handles[f"{builder_name}_{register_method}_{n}"] = handle

    def before_iter(self, runner):
        if self.every_n_inner_iters(runner, self.intervals):
            self.enabled_at_this_iter = True
        else:
            self.enabled_at_this_iter = False
            for k in self.enable_reg.keys():
                self.enable_reg[k] = False

    def update_hooks(self, quant_mode, force=False):
        activated_builders = []
        if self.enabled_at_this_iter or force:
            for name, builder in self.named_builders.items():
                activated = builder.inject_at(quant_mode)
                self.enable_reg[id(builder)] = activated
                if activated:
                    activated_builders.append(builder)
        if self.is_ddp:
            return activated_builders

    def add_builder(self, builder_cfg, enabled=False, model=None):
        builder_name = builder_cfg["name"]
        builder_cls = builder_cfg["type"]
        builder_args = builder_cfg["args"]
        need_reg = builder_args.get("need_reg", False)
        if need_reg:
            hook_reg = OrderedDict()
            self.hook_regs[builder_name] = hook_reg
            builder_args.pop("need_reg")
        else:
            hook_reg = None
        if builder_cls not in BUILDERS:
            builder_cls += "Builder"
        hook_builder = BUILDERS[builder_cls](hook_reg=hook_reg, enable_reg=self.enable_reg, **builder_args)
        self.named_builders[builder_name] = hook_builder
        self.enable_reg[id(hook_builder)] = enabled
        if model is not None:
            for n, m in model.named_modules():
                if hook_builder.match(n, m):
                    for register_method, hook in hook_builder.get_hooks():
                        handle = getattr(m, register_method)(hook)
                        self.named_handles[f"{builder_name}_{register_method}_{n}"] = handle
        return builder_name

    def remove_builder(self, builder_name):
        handles = []
        for k, h in self.named_handles.items():
            if k.startswith(builder_name):
                handles.append((k, h))
        for k, h in handles:
            h.remove()
            self.named_handles.pop(k)
        if builder_name in self.named_builders:
            builder = self.named_builders.pop(builder_name)
            self.enable_reg.pop(id(builder))
        if builder_name in self.hook_regs:
            self.hook_regs.pop(builder_name)


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

    def after_iter(self, runner):
        if self.enabled_at_this_iter:
            plot_buffer = OrderedDict()
            plot_method = OrderedDict()
            for name, process in self.post_process.items():
                result = process.after_iter(self.hook_regs, runner.outputs)
                plot_buffer[name] = result
                plot_method[name] = process.plot_method
            for _, reg in self.hook_regs.items():
                reg.clear()
            runner.log_buffer.output["plot_buffer"] = plot_buffer
            runner.log_buffer.output["plot_method"] = plot_method
