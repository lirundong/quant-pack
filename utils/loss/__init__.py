# -*- coding: utf-8 -*-

import torch.nn as nn

from .distillation_loss import *
from .inv_distillation_loss import *

__all__ = ["get_loss"]

_loss_zoo = {
    "KDistLoss": KDistLoss,
    "InvDistilLoss": InvDistilLoss,
}


def get_loss(name, *args, **kwargs):
    if name in _loss_zoo:
        loss_module = _loss_zoo[name]
    elif name in vars(nn):
        loss_module = vars(nn)[name]
    else:
        raise KeyError(f"unsupported loss function: {name}")

    return loss_module(*args, **kwargs)
