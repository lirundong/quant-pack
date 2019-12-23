# -*- coding: utf-8 -*-

import torch


def get_scalar(x):
    if torch.is_tensor(x):
        assert x.numel() == 1
        return x.item()
    else:
        return x
