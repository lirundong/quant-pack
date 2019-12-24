# -*- coding: utf-8 -*-

from collections import OrderedDict

import torch
import torch.distributed as dist

__all__ = ["cls_acc", "clear_or_init", "SyncValue"]


@torch.no_grad()
def cls_acc(logits, label, topk=(1,)):
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


def clear_or_init(obj, name):
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

