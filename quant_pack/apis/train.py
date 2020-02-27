# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader, DistributedSampler

import quant_pack.core.wrapper as wrapper
import quant_pack.core.runner as runner
from quant_pack.datasets import build_dataset
from quant_pack.models import build_model

from .utils import load_pre_trained, fresh_resume, item_to_tuple


def _dist_train(cfg):
    train_set, eval_set = build_dataset(cfg.dataset.name, eval_only=False, **cfg.dataset.args)
    train_loader = DataLoader(train_set, sampler=DistributedSampler(train_set), **cfg.train.data_loader.args)
    eval_loader = DataLoader(eval_set, sampler=DistributedSampler(eval_set), **cfg.eval.data_loader.args)

    model = build_model(cfg.model)
    if cfg.pre_trained:
        load_pre_trained(model, cfg.pre_trained)
    bn_folding_mapping = wrapper.track_bn_folding_mapping(model, torch.randn(*cfg.model.input_size))
    model = wrapper.__dict__[cfg.wrapper.name](model, bn_folding_mapping=bn_folding_mapping, **cfg.wrapper.args)
    model.module.to(cfg.device)

    optims = model.get_optimizers(*cfg.train.optim_groups)
    trainer = runner.MultiOptimRunner(model, model.batch_processor, optims, cfg.work_dir)
    trainer.register_qat_hooks(cfg.train.loss, cfg.train.metrics, cfg.train.lr_policies,
                               cfg.train.qat_policies, cfg.train.ckpt_interval, cfg.runtime_hooks)

    if cfg.eval:
        trainer.register_eval_hooks(cfg.eval.metrics)
    if cfg.log:
        trainer.register_logger_hooks(cfg.log)
    if cfg.resume:
        trainer.resume(cfg.resume)

    trainer.model.to_ddp()
    trainer.run([train_loader, eval_loader], item_to_tuple(*cfg.work_flow), cfg.epochs,
                device=cfg.device, runtime_hook=trainer.runtime_hook)


def _local_train(cfg):
    train_set, eval_set = build_dataset(cfg.dataset.name, eval_only=False, **cfg.dataset.args)
    train_loader = DataLoader(train_set, **cfg.train.data_loader.args)
    eval_loader = DataLoader(eval_set, **cfg.eval.data_loader.args)

    model = build_model(cfg.model)
    if cfg.pre_trained:
        load_pre_trained(model, cfg.pre_trained)
    bn_folding_mapping = wrapper.track_bn_folding_mapping(model, torch.randn(*cfg.model.input_size))
    model = wrapper.__dict__[cfg.wrapper.name](model, bn_folding_mapping=bn_folding_mapping, **cfg.wrapper.args)
    if cfg.fresh_resume:
        fresh_resume(model.module, cfg.fresh_resume)
    model.module.to(cfg.device)

    optims = model.get_optimizers(*cfg.train.optim_groups)
    trainer = runner.MultiOptimRunner(model, model.batch_processor, optims, cfg.work_dir)
    trainer.register_qat_hooks(cfg.train.loss, cfg.train.metrics, cfg.train.lr_policies,
                               cfg.train.qat_policies, cfg.train.ckpt_interval, cfg.runtime_hooks)

    if cfg.eval:
        trainer.register_eval_hooks(cfg.eval.metrics)
    if cfg.log:
        trainer.register_logger_hooks(cfg.log)
    if cfg.resume:
        trainer.resume(cfg.resume)

    trainer.run([train_loader, eval_loader], item_to_tuple(*cfg.work_flow), cfg.epochs,
                device=cfg.device, runtime_hook=trainer.runtime_hook)


def train_classifier(cfg):
    if cfg.distributed:
        _dist_train(cfg)
    else:
        _local_train(cfg)
