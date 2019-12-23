# -*- coding: utf-8 -*-

from . import lr_policies
from . import qat_policies
from . import cls_loss
from . import cls_metric

__all__ = ["build_qat_policies", "build_lr_policies", "build_loss", "build_metrics"]


def build_qat_policies(*cfgs):
    ret_policies = []
    for cfg in cfgs:
        policy = qat_policies.__dict__[cfg["name"]](**cfg["args"])
        ret_policies.append(policy)
    return ret_policies


def build_lr_policies(*cfgs):
    ret_policies = []
    for cfg in cfgs:
        lr_policy_cls = cfg["name"]
        if lr_policy_cls not in lr_policies.__dict__:
            lr_policy_cls += "LrUpdateHook"
        policy = lr_policies.__dict__[lr_policy_cls](**cfg["args"])
        ret_policies.append(policy)
    return ret_policies


def build_loss(cfg):
    loss_cls = cfg["name"]
    if loss_cls not in cls_loss.__dict__:
        loss_cls += "Loss"
    loss_hook = cls_loss.__dict__[loss_cls](**cfg["args"])
    return loss_hook


def build_metrics(*cfgs):
    ret_metrics = []
    for cfg in cfgs:
        metric_cls = cfg["name"]
        if metric_cls not in cls_metric.__dict__:
            metric_cls += "Metric"
        metric = cls_metric.__dict__[metric_cls](**cfg["args"])
        ret_metrics.append(metric)
    return ret_metrics
