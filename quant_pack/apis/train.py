# -*- coding: utf-8 -*-

import torch
import torchvision
from torch.utils.data import DataLoader, DistributedSampler

import quant_pack.core.wrapper as wrapper
import quant_pack.core.runner as runner
from quant_pack.datasets import get_dataset

__all__ = ["train_classifier"]


def _load_pre_trained(model, ckpt_path):
    device = torch.device("cpu")
    ckpt = torch.load(open(ckpt_path, "rb"), device)
    if "model" in ckpt.keys():
        ckpt = ckpt["model"]
    model.load_state_dict(ckpt)


def _dist_train(cfg):
    train_set, eval_set = get_dataset(cfg.dataset.name, eval_only=False, **cfg.dataset.args)
    train_loader = DataLoader(train_set, sampler=DistributedSampler(train_set), **cfg.train.data_loader.args)
    eval_loader = DataLoader(eval_set, sampler=DistributedSampler(eval_set), **cfg.eval.data_loader.args)

    model = torchvision.models.__dict__[cfg.model.name](**cfg.model.args)
    if cfg.pre_trained.before_bn_folding:
        _load_pre_trained(model, cfg.pre_trained.before_bn_folding)
    bn_folding_mapping = wrapper.track_bn_folding_mapping(model, torch.randn(*cfg.model.input_size))
    model = wrapper.__dict__[cfg.wrapper.name](model, bn_folding_mapping=bn_folding_mapping, **cfg.wrapper.args)
    if cfg.pre_trained.after_bn_folding:
        _load_pre_trained(model.module, cfg.pre_trained.after_bn_folding)
    model.module.to(cfg.device)
    model.to_ddp()

    optims = model.get_optimizers(cfg.train.optim_groups)
    trainer = runner.MultiOptimRunner(model, model.batch_processor, optims)
    trainer.register_qat_hooks(cfg.train.loss, cfg.train.lr_policies, cfg.train.qat_policies)
    trainer.register_eval_hooks(*cfg.eval.metrics)

    if cfg.resume:
        trainer.resume(cfg.resume)

    trainer.run([train_loader, eval_loader], cfg.work_flow, cfg.epochs)


def _local_train(cfg):
    raise NotImplementedError()


def train_classifier(cfg):
    if cfg.distributed:
        _dist_train(cfg)
    else:
        _local_train(cfg)
