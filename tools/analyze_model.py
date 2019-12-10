# -*- coding: utf-8 -*-

from argparse import ArgumentParser
from collections import OrderedDict
from pprint import pformat

import torch
import torch.nn as nn

from quant_pack import modeling

_tasks = {}


def register(name):
    def _do_register(func):
        assert callable(func)
        _tasks[name] = func
        return func
    return _do_register


@register("mag")
def weight_channel_magnitudes_ratio(model: nn.Module) -> OrderedDict:
    ret = OrderedDict()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "weight" in n and p.dim() > 1:
                c = p.size(0)
                p_max, _ = p.view(c, -1).max(dim=1)
                p_min, _ = p.view(c, -1).min(dim=1)
                p_mag = p_max - p_min
                ret[n] = (p_mag.max() / p_mag.min()).item()

    return ret


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", help="path to model checkpoint")
    parser.add_argument("-a", "--arch", default="resnet18_idq", help="model name")
    parser.add_argument("-t", "--task", choices=("mag",), help="analysis type")
    args = parser.parse_args()

    device = torch.device("cpu")
    ckpt = torch.load(args.model, device)
    if "model" in ckpt.keys():
        ckpt = ckpt["model"]
    model = modeling.__dict__[args.arch]()
    model.load_state_dict(ckpt, strict=False)
    f = _tasks[args.task]
    ret = f(model)

    print(pformat(ret))
