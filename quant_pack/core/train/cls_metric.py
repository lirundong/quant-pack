# -*- coding: utf-8 -*-

from mmcv.runner import Hook

from quant_pack.core.utils import cls_acc

from .utils import get_scalar


class TopKMetric(Hook):

    def __init__(self, logits_names, topk):
        self.logits_names = logits_names
        self.topk = topk

    def after_train_iter(self, runner):
        log_vars = {}
        for name in self.logits_names:
            if name in runner.outputs:
                logits = runner.outputs[name]
                topk_acc = cls_acc(logits, runner.outputs["label"], self.topk)
                for k, acc in zip(self.topk, topk_acc):
                    log_vars[f"{name}_top{k}_train_acc"] = get_scalar(acc)
        runner.log_buffer.update(log_vars)
