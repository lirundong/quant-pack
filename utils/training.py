# -*- coding: utf-8 -*-

import logging
from collections import OrderedDict
from copy import deepcopy
from collections import deque
from datetime import timedelta
from typing import Tuple, List

import torch
import torch.nn as nn
import numpy as np
import linklink as link

TensorT = torch.Tensor
IListT = Tuple[int], List[int]
RListT = Tuple[float], List[float]

_rank = 0
__all__ = ["init_log", "accuracy", "map_to_cpu", "get_eta", "update_bn_stat",
           "param_grad_ratio", "AverageMeter"]


def init_log(debug=False, rank=0):
    global _rank
    _rank = rank
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
    logger.addFilter(lambda record: _rank == 0)

    return logger


def accuracy(output: TensorT, target: TensorT, world_size: int = 1,
             debug: bool = False, topk: IListT = (1, )) -> RListT:
    """Computes the accuracy over the k top predictions for the specified values of k"""
    if output is None:
        return [0.0, ] * len(topk)
    accs = []
    maxk = max(*topk, 2)
    with torch.no_grad():
        if not torch.is_tensor(output):
            output = output[-1]

        batch_size = torch.tensor(target.size(0), device=output.device).float()
        if world_size > 1:
            link.allreduce(batch_size)

        top_logits, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
        if debug:
            logits_diff = top_logits[:, 0] - top_logits[:, 1]
            diff_large_top, _ = logits_diff.topk(3, largest=True)
            diff_small_top, _ = logits_diff.topk(3, largest=False)
            logger = logging.getLogger("global")
            logger.debug(f"Top-3 logits diff: {diff_large_top}")
            logger.debug(f"Bottom-3 logits diff: {diff_small_top}")
        pred = pred.t()  # shape: (k, N)
        correct = pred.eq(target.view(1, -1).expand_as(pred)).float()
        if world_size > 1:
            link.allreduce(correct)

        for k in topk:
            correct_k = correct[:k].view(-1).sum(0, keepdim=True)
            acc = correct_k.mul_(100.0).div_(batch_size)
            accs.append(acc.item())
    return accs


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


def param_grad_ratio(model):
    ratio_dict = OrderedDict()
    eps = 1e-5
    with torch.no_grad():
        for n, p in model.named_parameters():
            if p.grad is not None:
                g_norm = p.grad.norm()
                p_norm = p.norm()
                ratio_dict[n] = g_norm.div(p_norm.add(eps)).item()

    return ratio_dict


class AverageMeter:
    def __init__(self, memo=50):
        self.memo = deque(maxlen=memo)

    def set(self, val):
        self.memo.append(val)

    def val(self):
        return self.memo[-1]

    def avg(self):
        return np.mean(self.memo)
