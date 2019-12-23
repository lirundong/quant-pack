# -*- coding: utf-8 -*-

import torch


def load_pre_trained(model, ckpt_path):
    device = torch.device("cpu")
    ckpt = torch.load(open(ckpt_path, "rb"), device)
    if "model" in ckpt.keys():
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)


def item_to_tuple(*args):
    ret = []
    for arg in args:
        assert isinstance(arg, (tuple, list))
        ret.append(tuple(arg))
    return ret
