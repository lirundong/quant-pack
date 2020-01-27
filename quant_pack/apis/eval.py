# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader, DistributedSampler

import quant_pack.core.wrapper as wrapper
import quant_pack.core.runner as runner
from quant_pack.datasets import build_dataset
from quant_pack.models import build_model

from .utils import load_pre_trained


def _dist_eval(cfg):
    eval_set = build_dataset(cfg.dataset.name, eval_only=True, **cfg.dataset.args)
    eval_loader = DataLoader(eval_set, sampler=DistributedSampler(eval_set), **cfg.eval.data_loader.args)

    model = build_model(cfg.model)
    if cfg.pre_trained:
        load_pre_trained(model, cfg.pre_trained)
    bn_folding_mapping = wrapper.track_bn_folding_mapping(model, torch.randn(*cfg.model.input_size))
    model = wrapper.__dict__[cfg.wrapper.name](model, bn_folding_mapping=bn_folding_mapping, **cfg.wrapper.args)
    model.module.to(cfg.device)

    evaluator = runner.MultiOptimRunner(model, model.batch_processor, work_dir=cfg.work_dir)
    evaluator.register_eval_hooks(cfg.eval.metrics)

    if cfg.runtime_hooks:
        evaluator.inject_runtime_hooks(**cfg.runtime_hooks)
    if cfg.log:
        evaluator.register_logger_hooks(cfg.log)
    if cfg.resume:
        evaluator.resume(cfg.resume, resume_optimizer=False)

    model.to_ddp()
    evaluator.call_hook("before_run")
    evaluator.val(eval_loader, device=cfg.device, quant_mode=cfg.eval.quant_mode)


def _local_eval(cfg):
    eval_set = build_dataset(cfg.dataset.name, eval_only=True, **cfg.dataset.args)
    eval_loader = DataLoader(eval_set, **cfg.eval.data_loader.args)

    model = build_model(cfg.model)
    if cfg.pre_trained:
        load_pre_trained(model, cfg.pre_trained)
    bn_folding_mapping = wrapper.track_bn_folding_mapping(model, torch.randn(*cfg.model.input_size))
    model = wrapper.__dict__[cfg.wrapper.name](model, bn_folding_mapping=bn_folding_mapping, **cfg.wrapper.args)
    model.module.to(cfg.device)

    evaluator = runner.MultiOptimRunner(model, model.batch_processor, work_dir=cfg.work_dir)
    evaluator.register_eval_hooks(cfg.eval.metrics)

    if cfg.runtime_hooks:
        runtime_hook_updater = evaluator.inject_runtime_hooks(**cfg.runtime_hooks)
    else:
        runtime_hook_updater = None
    if cfg.log:
        evaluator.register_logger_hooks(cfg.log)
    if cfg.resume:
        evaluator.resume(cfg.resume, resume_optimizer=False)

    evaluator.call_hook("before_run")
    evaluator.val(eval_loader, device=cfg.device, quant_mode=cfg.eval.quant_mode, runtime_hook_updater=runtime_hook_updater)


def eval_classifier(cfg):
    if cfg.distributed:
        _dist_eval(cfg)
    else:
        _local_eval(cfg)
