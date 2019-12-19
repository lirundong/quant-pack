# -*- coding: utf-8 -*-

from . import lr_policies
from . import qat_policies
from . import loss

__all__ = ["build_qat_policies", "build_lr_policies", "build_losses"]


def build_qat_policies(*cfgs):
    ret_policies = []
    for cfg in cfgs:
        policy = qat_policies.__dict__[cfg["name"]](**cfg["args"])
        ret_policies.append(policy)
    return ret_policies


def build_lr_policies(*cfgs):
    ret_policies = []
    for cfg in cfgs:
        name = cfg["name"]
        if name not in lr_policies.__dict__.keys():
            name += "LrUpdateHook"
        policy = lr_policies.__dict__[name](**cfg["args"])
        ret_policies.append(policy)
    return ret_policies


def build_losses(cfg):
    loss_cls = cfg["name"]
    loss_hook = loss.__dict__[loss_cls](**cfg["args"])
    return loss_hook
