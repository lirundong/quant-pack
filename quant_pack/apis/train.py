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


def _item_to_tuple(*args):
    ret = []
    for arg in args:
        assert isinstance(arg, (tuple, list))
        ret.append(tuple(arg))
    return ret


def _dist_train(cfg):
    train_set, eval_set = get_dataset(cfg.dataset.name, eval_only=False, **cfg.dataset.args)
    train_loader = DataLoader(train_set, sampler=DistributedSampler(train_set), **cfg.train.data_loader.args)
    eval_loader = DataLoader(eval_set, sampler=DistributedSampler(eval_set), **cfg.eval.data_loader.args)

    model = torchvision.models.__dict__[cfg.model.name](**cfg.model.args)
    if cfg.pre_trained:
        _load_pre_trained(model, cfg.pre_trained)
    bn_folding_mapping = wrapper.track_bn_folding_mapping(model, torch.randn(*cfg.model.input_size))
    model = wrapper.__dict__[cfg.wrapper.name](model, bn_folding_mapping=bn_folding_mapping, **cfg.wrapper.args)
    model.module.to(cfg.device)
    model.to_ddp(cfg.device)

    optims = model.get_optimizers(*cfg.train.optim_groups)
    trainer = runner.MultiOptimRunner(model, model.batch_processor, optims, cfg.work_dir)
    trainer.register_qat_hooks(cfg.train.loss, cfg.train.metrics, cfg.train.lr_policies, cfg.train.qat_policies)

    if cfg.runtime_hooks:
        trainer.inject_runtime_hooks(cfg.runtime_hooks)
    if cfg.eval:
        trainer.register_eval_hooks(cfg.eval.metrics)
    if cfg.log:
        trainer.register_logger_hooks(cfg.log)
    if cfg.resume:
        trainer.resume(cfg.resume)

    trainer.run([train_loader, eval_loader], _item_to_tuple(*cfg.work_flow), cfg.epochs, device=cfg.device)


def _local_train(cfg):
    raise NotImplementedError()


def train_classifier(cfg):
    if cfg.distributed:
        _dist_train(cfg)
    else:
        _local_train(cfg)
