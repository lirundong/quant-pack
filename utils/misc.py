# -*- coding: utf-8 -*-

import os
from collections import OrderedDict
from copy import deepcopy
from datetime import timedelta

import torch

__all__ = ["accuracy", "get_eta", "update_config", "Checkpointer"]


def accuracy(output, target, topk=(1,)):
    if output is None:
        assert torch.is_tensor(target)
        return [torch.tensor(0., device=target.device), ] * len(topk)

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target[None])

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().sum(dtype=torch.float32)
            res.append(correct_k * (100.0 / batch_size))
        return res


def get_eta(gone_steps, total_steps, speed):
    remain_steps = total_steps - gone_steps
    remain_seconds = remain_steps * speed

    return timedelta(seconds=remain_seconds)


def update_config(conf: dict, extra: dict) -> None:

    def _update_item(c, k, v):
        if "." in k:
            tokens = k.split(".")
            current_k, remain_k = tokens[0], ".".join(tokens[1:])
            c.setdefault(current_k, dict())
            _update_item(c[current_k], remain_k, v)
        else:
            c[k] = v
            return

    for k, v in extra.items():
        _update_item(conf, k, v)


class Checkpointer:

    def __init__(self, ckpt_dir, rank=None):
        if rank is None:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0
        self.is_master = rank == 0
        if not self.is_master:
            return
        self.registry = OrderedDict()
        self.ckpt_dir = ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)

    @staticmethod
    def _copy_to_cpu(**kwargs):
        cpu = torch.device("cpu")
        cpu_dict = OrderedDict()
        for k, v in kwargs.items():
            if torch.is_tensor(v):
                cpu_dict[k] = deepcopy(v.to(cpu))
            else:
                cpu_dict[k] = deepcopy(v)
        return cpu_dict

    def save(self, **kwargs):
        if not self.is_master:
            return
        cpu_dict = self._copy_to_cpu(**kwargs)
        self.registry.update(cpu_dict)

    def write_to_disk(self, fname):
        if not self.is_master or len(self.registry) == 0:
            return
        ckpt_path = os.path.join(self.ckpt_dir, fname)
        with open(ckpt_path, "wb") as f:
            torch.save(self.registry, f)
