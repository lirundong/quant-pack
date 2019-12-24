# -*- coding: utf-8 -*-

import torch.distributed as dist
from mmcv.runner import Hook

from quant_pack.core.utils import cls_acc, SyncValue, clear_or_init


class DistEvalTopKHook(Hook):

    BUFFER_NAME = "val_buffer"

    def __init__(self, logits_names, topk):
        assert dist.is_available() and dist.is_initialized()
        self.logits_names = logits_names
        self.topk = topk

    def before_val_epoch(self, runner):
        clear_or_init(runner, self.BUFFER_NAME)
        buffer = getattr(runner, self.BUFFER_NAME)
        for name in self.logits_names:
            for k in self.topk:
                buffer[f"{name}_top{k}_eval_acc"] = SyncValue()

    def after_val_iter(self, runner):
        buffer = getattr(runner, self.BUFFER_NAME)
        for name in self.logits_names:
            if name in runner.outputs:
                logits = runner.outputs[name]
                batch_size = logits.size(0)
                topk = cls_acc(logits, runner.outputs["label"], self.topk)
                for k, acc in zip(self.topk, topk):
                    buffer[f"{name}_top{k}_eval_acc"].update(acc, n=batch_size)

    def after_val_epoch(self, runner):
        device = next(iter(runner.model.parameters())).device
        buffer = getattr(runner, self.BUFFER_NAME)
        log_vars = {}
        for name, val in buffer.items():
            if val.count > 0:
                val.synchronize_between_processes(device)
                log_vars[name] = val.global_avg
        runner.log_buffer.output.update(log_vars)
