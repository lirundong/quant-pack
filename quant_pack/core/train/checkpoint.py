# -*- coding: utf-8 -*-

import signal
import os
import time
from io import BytesIO

import colorama
import mmcv
import torch
from mmcv.runner import Hook, master_only, weights_to_cpu


def save_checkpoint(model, file, optimizer=None, meta=None):
    if meta is None:
        meta = {}
    meta.update(mmcv_version=mmcv.__version__, time=time.asctime())
    if hasattr(model, 'module'):
        model = model.module
    checkpoint = {
        'meta': meta,
        'state_dict': weights_to_cpu(model.state_dict())
    }
    if optimizer is not None:
        checkpoint['optimizer'] = optimizer.state_dict()
    torch.save(checkpoint, file)


class RAMBufferedCheckpointHook(Hook):

    def __init__(self, criterion, interval=-1, save_optimizer=True, output_dir=None, **kwargs):
        self.criterion = criterion
        self.interval = interval
        self.save_optimizer = save_optimizer
        self.output_dir = output_dir
        self.kwargs = kwargs
        self.best_acc = 0.0
        self.best_ckpt = None
        self.latest_ckpt = None
        signal.signal(signal.SIGINT, lambda sig, frame: RAMBufferedCheckpointHook._handle_sigint(self, sig, frame))
        signal.signal(signal.SIGTERM, lambda sig, frame: RAMBufferedCheckpointHook._handle_sigint(self, sig, frame))

    @staticmethod
    def _handle_sigint(self, sig, frame):
        print(colorama.Fore.YELLOW + f"\nWaite a minute, writing checkpoints to disk...", flush=True)
        self.write_to_disk()
        print(colorama.Fore.GREEN + f"Checkpoints have been writen to: {self.output_dir}", flush=True)
        print(colorama.Style.RESET_ALL, flush=True)
        exit(0)

    @master_only
    def write_to_disk(self):
        assert self.latest_ckpt is not None and self.best_ckpt is not None
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        with open(os.path.join(self.output_dir, "ckpt_latest.pth"), "wb") as f:
            f.write(self.latest_ckpt.getbuffer())
        with open(os.path.join(self.output_dir, "ckpt_best.pth"), "wb") as f:
            f.write(self.best_ckpt.getbuffer())

    @master_only
    def after_val_epoch(self, runner):
        if self.output_dir is None:
            self.output_dir = runner.work_dir
        if isinstance(self.criterion, (list, tuple)):
            criterion = next(c for c in self.criterion if c in runner.log_buffer.output)
        else:
            criterion = self.criterion
        acc = runner.log_buffer.output[criterion]
        meta = dict(epoch=runner.epoch + 1, iter=runner.iter, **self.kwargs)
        meta[criterion] = acc
        optimizer = runner.optimizer if self.save_optimizer else None
        f = BytesIO()
        save_checkpoint(runner.model, f, optimizer, meta)
        self.latest_ckpt = f
        if acc > self.best_acc or self.best_ckpt is None:
            self.best_acc = acc
            self.best_ckpt = f
        if self.every_n_epochs(runner, self.interval):
            self.write_to_disk()

    @master_only
    def after_run(self, runner):
        self.write_to_disk()
