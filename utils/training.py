# -*- coding: utf-8 -*-

import logging
from collections import OrderedDict
from copy import deepcopy
from collections import deque
from datetime import timedelta

import torch
import torch.nn as nn
import numpy as np


__all__ = ["init_log", "accuracy", "map_to_cpu", "get_eta", "update_bn_stat",
           "AverageMeter"]


def init_log(debug=False):
    level = logging.DEBUG if debug else logging.INFO
    log_fmt = f"%(asctime)s-%(filename)s#%(lineno)d: [%(levelname)s] %(message)s"
    date_fmt = "%Y-%m-%d_%H:%M:%S"
    fmt = logging.Formatter(log_fmt, date_fmt)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger = logging.getLogger("global")
    logger.setLevel(level)
    logger.addHandler(ch)

    return logger


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


def update_bn_stat(model, loader):

    def update_bn_stat_pre_hook(m, input):
        assert isinstance(m, nn.BatchNorm2d)
        with torch.no_grad():
            x, = input
            c = x.size(1)
            x = x.permute(1, 0, 2, 3).reshape(c, -1)
            x_mean = x.mean(dim=1)
            x_var = x.var(dim=1)
            m.running_mean = x_mean
            m.running_var = x_var

    model.eval()
    device = next(model.parameters()).device
    hooks = []
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            h = m.register_forward_pre_hook(update_bn_stat_pre_hook)
            hooks.append(h)

    with torch.no_grad():
        for i, (img, _) in enumerate(loader):
            if i > 0:
                break
            img = img.to(device, non_blocking=True)
            logitis = model(img)

    for h in hooks:
        h.remove()


class AverageMeter:
    def __init__(self, memo=50):
        self.memo = deque(maxlen=memo)

    def set(self, val):
        self.memo.append(val)

    def val(self):
        return self.memo[-1]

    def avg(self):
        return np.mean(self.memo)
