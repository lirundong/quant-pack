# -*- coding: utf-8 -*-

import torch
import torchvision
from torch.utils.data import DataLoader, DistributedSampler

import quant_pack.core.wrapper as wrapper
import quant_pack.core.runner as runner
from quant_pack.datasets import get_dataset

from .utils import load_pre_trained


def _dist_eval(cfg):
    eval_set = get_dataset(cfg.dataset.name, eval_only=True, **cfg.dataset.args)
    eval_loader = DataLoader(eval_set, sampler=DistributedSampler(eval_set), **cfg.eval.data_loader.args)

    model = torchvision.models.__dict__[cfg.model.name](**cfg.model.args)
    if cfg.pre_trained:
        load_pre_trained(model, cfg.pre_trained)
    bn_folding_mapping = wrapper.track_bn_folding_mapping(model, torch.randn(*cfg.model.input_size))
    model = wrapper.__dict__[cfg.wrapper.name](model, bn_folding_mapping=bn_folding_mapping, **cfg.wrapper.args)
    model.module.to(cfg.device)
    model.to_ddp(cfg.device)

    evaluator = runner.MultiOptimRunner(model, model.batch_processor, work_dir=cfg.work_dir)
    evaluator.register_eval_hooks(cfg.eval.metrics)

    if cfg.runtime_hooks:
        evaluator.inject_runtime_hooks(cfg.runtime_hooks)
    if cfg.log:
        evaluator.register_logger_hooks(cfg.log)
    if cfg.resume:
        evaluator.resume(cfg.resume, resume_optimizer=False)

    evaluator.call_hook("before_run")
    evaluator.val(eval_loader, device=cfg.device, quant_mode=cfg.eval.quant_mode)


def _local_eval(cfg):
    raise NotImplementedError()


def eval_classifier(cfg):
    if cfg.distributed:
        _dist_eval(cfg)
    else:
        _local_eval(cfg)
