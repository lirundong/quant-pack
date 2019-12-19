# -*- coding: utf-8 -*-

from collections import OrderedDict

import torch
import torch.distributed as dist
from mmcv.runner import Hook

__all__ = ["DistEvalTopKHook"]


@torch.no_grad()
def accuracy(logits, label, topk=(1,)):
    assert torch.is_tensor(logits) and torch.is_tensor(label)
    maxk = max(topk)
    batch_size = label.size(0)
    res = []
    _, pred = torch.topk(logits, maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(label[None])
    for k in topk:
        correct_k = correct[:k].flatten().sum(dtype=torch.float32)
        res.append(correct_k * (100.0 / batch_size))
    return res


def check_and_init(obj, name):
    if not hasattr(obj, name):
        setattr(obj, name, OrderedDict())
    else:
        getattr(obj, name).clear()


class SyncValue:

    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, value, n=1):
        if torch.is_tensor(value):
            assert value.numel() == 1
            value = value.item()
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self, device):
        if not (dist.is_available() and dist.is_initialized()):
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device=device)
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def global_avg(self):
        return self.total / self.count


class DistEvalTopKHook(Hook):

    BUFFER_NAME = "val_buffer"
    RESULTS_NAME = "val_results"

    def __init__(self, logits_names, topk):
        assert dist.is_available() and dist.is_initialized()
        self.logits_names = logits_names
        self.topk = topk

    def before_val_epoch(self, runner):
        check_and_init(runner, self.BUFFER_NAME)
        buffer = getattr(runner, self.BUFFER_NAME)
        for name in self.logits_names:
            for k in self.topk:
                buffer[f"{name}_top_{k}"] = SyncValue()

    def after_val_iter(self, runner):
        for name in self.logits_names:
            logits = runner.outputs[name]
            batch_size = logits.size(0)
            topk = accuracy(logits, runner.outputs["label"], self.topk)
            for k, acc in zip(self.topk, topk):
                runner.val_buffer[f"{name}_top_{k}"].update(acc, n=batch_size)

    def after_val_epoch(self, runner):
        check_and_init(runner, self.RESULTS_NAME)
        results = getattr(runner, self.RESULTS_NAME)
        buffer = getattr(runner, self.BUFFER_NAME)
        results["epoch"] = runner.epoch
        for name, val in buffer.items():
            val.synchronize_between_processes(runner.device)
            results[name] = val.global_avg
