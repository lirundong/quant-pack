# -*- coding: utf-8 -*-

import os
import signal
from io import BytesIO
from glob import glob
from datetime import timedelta
from pathlib import Path
from deepmerge import Merger

import torch
import yaml

__all__ = ["accuracy", "get_eta", "get_latest_file", "update_config", "Checkpointer"]


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    assert torch.is_tensor(target)
    if torch.is_tensor(output):
        output = (output, )
    maxk = max(topk)
    batch_size = target.size(0)
    results = []
    for logits in output:
        if logits is None:
            res = [torch.tensor(0., device=target.device), ] * len(topk)
        else:
            res = []
            _, pred = logits.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target[None])
            for k in topk:
                correct_k = correct[:k].flatten().sum(dtype=torch.float32)
                res.append(correct_k * (100.0 / batch_size))
        results.append(res)

    return results


def get_eta(gone_steps, total_steps, speed):
    remain_steps = total_steps - gone_steps
    remain_seconds = remain_steps * speed

    return timedelta(seconds=remain_seconds)


def get_latest_file(path: str) -> str:
    files = glob(os.path.join(path, "*"))
    latest = sorted(files, key=os.path.getmtime)
    return latest[-1]


def update_config(conf: dict, extra: dict) -> dict:

    def _update_item(c, k, v):
        if "." in k:
            tokens = k.split(".")
            current_k, remain_k = tokens[0], ".".join(tokens[1:])
            c.setdefault(current_k, dict())
            _update_item(c[current_k], remain_k, v)
        else:
            c[k] = v
            return

    if "__BASE__" in conf.keys():
        base_path = conf["__BASE__"]
        conf.pop("__BASE__")
        if not os.path.exists(base_path):
            # __file__: quant-prob/quant_prob/utils/misc.py
            project_root = Path(__file__).absolute().parents[2]
            base_path = os.path.join(project_root, base_path)
        base_conf = yaml.load(open(base_path, "r", encoding="utf-8"),
                              Loader=yaml.SafeLoader)
    else:
        base_conf = {}

    # strategies for merging configs:
    #  - override lists in BASE by the counterparts in current config, such that
    #    each param-group can get the latest, non-duplicated config;
    #  - merge dicts recursively, such that derived config can be written as concisely as possible;
    conf_merger = Merger([(list, "override"), (dict, "merge")], ["override"], ["override"])
    conf_merger.merge(base_conf, conf)

    for k, v in extra.items():
        _update_item(base_conf, k, v)

    return base_conf


class Checkpointer:

    def __init__(self, ckpt_dir, rank=None):
        if rank is None:
            import torch.distributed as dist
            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
            else:
                rank = 0
        self.is_master = rank is 0
        if not self.is_master:
            return
        self.ckpt_dir = ckpt_dir
        self.best_acc = 0
        self.latest_ckpt = None
        self.best_ckpt = None
        os.makedirs(ckpt_dir, exist_ok=True)
        # bind `self` to `_handle_sigint`
        signal.signal(signal.SIGINT, lambda sig, frame: Checkpointer._handle_sigint(self, sig, frame))

    @staticmethod
    def _handle_sigint(self, sig, frame):
        print(f"\nwaite a minute, writing checkpoints to disk...", flush=True)
        self.write_to_disk()
        print(f"checkpoints have been writen to {self.ckpt_dir}", flush=True)
        exit(0)

    def save(self, step, accuracy, **kwargs):
        if not self.is_master:
            return
        f = BytesIO()
        torch.save({
            "step": step,
            "accuracy": accuracy,
            **kwargs,
        }, f)
        self.latest_ckpt = f
        if accuracy > self.best_acc:
            self.best_acc = accuracy
            self.best_ckpt = f

    def write_to_disk(self):
        if not self.is_master:
            return
        assert self.latest_ckpt is not None and self.best_ckpt is not None, \
            "did you call `save` before `write_to_dick`?"
        with open(os.path.join(self.ckpt_dir, "ckpt_latest.pth"), "wb") as f:
            f.write(self.latest_ckpt.getbuffer())
        with open(os.path.join(self.ckpt_dir, "ckpt_best.pth"), "wb") as f:
            f.write(self.best_ckpt.getbuffer())
