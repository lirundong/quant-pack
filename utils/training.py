# -*- coding: utf-8 -*-

from collections import OrderedDict
from copy import deepcopy
from collections import deque
from datetime import timedelta

import torch
import numpy as np


__all__ = ["accuracy", "map_to_cpu", "get_eta", "AverageMeter"]


def accuracy(output, target):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        correct_k = correct.view(-1).float().sum(0, keepdim=True)
        return correct_k.mul_(100.0 / batch_size).item()


def map_to_cpu(src):
    cpu = torch.device("cpu")
    dst = OrderedDict()
    for k, v in src.items():
        if torch.is_tensor(v):
            dst[k] = deepcopy(v.to(cpu))
        else:
            dst[k] = deepcopy(v)
    return dst


def get_eta(gone_steps, total_steps, speed):
    remain_steps = total_steps - gone_steps
    remain_seconds = remain_steps * speed

    return timedelta(seconds=remain_seconds)


class AverageMeter:
    def __init__(self, memo=50):
        self.memo = deque(maxlen=memo)

    def set(self, val):
        self.memo.append(val)

    def val(self):
        return self.memo[-1]

    def avg(self):
        return np.mean(self.memo)
