# -*- coding: utf-8 -*-

import torchvision.models as models

from . import resnet_cifar
from . import init_utils
from . import mobilenet_v1

_model_reg = {}
_model_reg.update(models.__dict__)
_model_reg.update(resnet_cifar.__dict__)
_model_reg.update(mobilenet_v1.__dict__)

__all__ = ["build_model"]


def build_model(cfg):
    model_cls = cfg["name"]
    model_args = cfg["args"]
    model = _model_reg[model_cls](**model_args)

    init_method = cfg["init"]
    init_args = cfg.get("init_args", {})
    if init_method not in init_utils.__dict__:
        init_method += "_init_"
    init_method_ = init_utils.__dict__[init_method]
    init_method_(model, **init_args)

    return model
