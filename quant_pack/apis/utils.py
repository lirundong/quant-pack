# -*- coding: utf-8 -*-

import torch
import torch.nn as nn


def load_pre_trained(model, ckpt_path):
    device = torch.device("cpu")
    ckpt = torch.load(open(ckpt_path, "rb"), device)
    if "model" in ckpt.keys():
        ckpt = ckpt["model"]

    # NOTE: BC, prev multi-domain BN implementation registered additional EMA buffers to BN layers
    from_prev_multi_bn = False
    for k in ckpt.keys():
        if "running_mean_q" in k or "running_var_q" in k:
            from_prev_multi_bn = True
            break
    if from_prev_multi_bn:
        for m in model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.register_buffer("running_mean_q", m.running_mean.clone())
                m.register_buffer("running_var_q", m.running_var.clone())
                m.register_buffer("num_batches_tracked_q", m.num_batches_tracked.clone())
    model.load_state_dict(ckpt, strict=True)


def fresh_resume(model, ckpt_path):
    device = torch.device("cpu")
    ckpt = torch.load(open(ckpt_path, "rb"), device)
    if "state_dict" in ckpt.keys():
        ckpt = ckpt["state_dict"]
    model.load_state_dict(ckpt, strict=False)


def item_to_tuple(*args):
    ret = []
    for arg in args:
        assert isinstance(arg, (tuple, list))
        ret.append(tuple(arg))
    return ret
