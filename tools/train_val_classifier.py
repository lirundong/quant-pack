# -*- coding: utf-8 -*-

import os
import logging
import json
from argparse import ArgumentParser
from time import time
from datetime import datetime
from copy import deepcopy
from pprint import pformat

import yaml
import torch
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tensorboardX import SummaryWriter
from easydict import EasyDict as edict
import linklink as link

import backbone
from utils import *


best_accuracy = 0.
exp_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
conf = None
device = None
tb_logger = None
rank = 0
world_size = 1


def main():
    global best_accuracy, conf, device, tb_logger, rank, world_size

    parser = ArgumentParser(f"Probabilistic quantization neural networks.")
    parser.add_argument("--conf-path", help="path of configuration file")
    parser.add_argument("--evaluate", "-e", action="store_true", help="evaluate trained model")
    parser.add_argument("--quant", "-q", action="store_true", help="evaluate trained model")
    parser.add_argument("--extra", "-x", type=json.loads, help="extra configurations in json format")
    parser.add_argument("--comment", "-m", help="comment for each experiment")
    parser.add_argument("--debug", action="store_true", help="logging debug info")
    args = parser.parse_args()

    with open(args.conf_path, "r", encoding="utf-8") as f:
        conf = yaml.load(f)
        conf.update(vars(args))
        if args.extra is not None:
            conf.update(args.extra)
        conf = edict(conf)

    rank, world_size = dist_init()
    conf.dist = world_size > 1
    logger = init_log(conf.debug, rank)
    device = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")

    logger.debug(f"configurations:\n{pformat(conf)}")
    logger.debug(f"device: {device}")

    logger.debug(f"building dataset {conf.dataset.name}...")
    train_set, val_set = get_dataset(conf.dataset.name, **conf.dataset.args)
    train_loader = DataLoader(train_set,
                              sampler=IterationSampler(train_set, rank=rank, world_size=world_size,
                                                       **conf.train_samper_conf),
                              **conf.train_data_conf)
    val_loader = DataLoader(val_set,
                            sampler=DistributedSampler(val_set, world_size, rank) if conf.dist else None,
                            **conf.val_data_conf)
    logger.debug(f"build dataset {conf.dataset.name} done")

    logger.debug(f"building model `{conf.arch}`...")
    model = backbone.__dict__[conf.arch](**conf.arch_conf).to(device, non_blocking=True)
    model.quant(conf.quant)
    logger.debug(f"build model {model.__class__.__name__} done:\n{model}")
    logger.debug(f"model quantization: {conf.quant}")

    optimizer = optim.__dict__[conf.opt](model.opt_param_groups(conf.opt_prob, conf.denoise_only, **conf.opt_conf),
                                         **conf.opt_conf)
    scheduler = IterationScheduler(optimizer, dataset_size=len(train_set), world_size=world_size, **conf.scheduler_conf)

    num_params = 0
    for p in optimizer.param_groups:
        num_params += len(p["params"])
    logger.debug(f"number of parameters: {num_params}")

    if "resume_path" in conf:
        logger.debug(f"loading checkpoint at: {conf.resume_path}...")
        checkpoint = torch.load(conf.resume_path, device)
        model_dict = checkpoint["model"] if "model" in checkpoint.keys() else checkpoint
        model.load_state_dict(model_dict, strict=False)
        if "reset_p" in conf and conf.reset_p:
            logger.debug(f"resetting probabilistic parameters...")
            model.reset_p()
        if "resume_opt" in conf and conf.resume_opt:
            logger.debug(f"recovering optimizer...")
            optimizer.load_state_dict(checkpoint["opt"])
            best_accuracy = checkpoint["accuracy"]
            scheduler.load_state_dict(checkpoint["scheduler"])
            logger.debug(f"recovered opt at iteration: {scheduler.last_iter}")

    if conf.distillation:
        logger.debug("building FP teacher model...")
        teacher = deepcopy(model)
        teacher.full_precision()
        teacher.to(device, non_blocking=True)
        model.register_teacher(teacher)
        criterion = KDistLoss(conf.soft_weight, conf.temperature)
    else:
        teacher = None
        criterion = nn.CrossEntropyLoss().to(device, non_blocking=True)

    if "tb_dir" in conf and rank == 0:
        tb_dir = os.path.join(conf.tb_dir, f"{exp_datetime}_{conf.comment}")
        logger.debug(f"creating TensorBoard at: {tb_dir}...")
        os.makedirs(tb_dir, exist_ok=True)
        tb_logger = SummaryWriter(tb_dir)
        model.register_vis(tb_logger, "quant")
        if teacher is not None:
            teacher.register_vis(tb_logger, "teacher")

    if conf.dist:
        logger.debug(f"register all_reduce gradient hooks to model...")
        model = get_dist_module(model)

    if conf.update_bn:
        logger.debug(f"updating BN statistics by unlabeled data")
        update_bn_stat(model, train_loader)

    if conf.evaluate:
        assert conf.resume_path is not None, f"load state_dict before evaluating"
        step = scheduler.last_iter
        logger.info(f"[Step {step}]: evaluating...")
        accuracy = evaluate(model, val_loader, step)
        logger.info(f"[Step {step}]: accuracy {accuracy:.3f}%.")
        return

    train(model, criterion, train_loader, val_loader, optimizer, scheduler, teacher)


def train(model, criterion, train_loader, val_loader, optimizer, scheduler,
          teacher_model=None):

    def barrier():
        if conf.dist:
            link.barrier()

    global best_accuracy
    logger = logging.getLogger("global")
    best_model = None
    t_iter = AverageMeter(20)
    train_acc = AverageMeter(20)
    model.train()
    train_loader.sampler.set_last_iter(scheduler.last_iter)

    t0 = time()
    for i, (img, label) in enumerate(train_loader):
        t_data = time() - t0
        scheduler.step()
        step = scheduler.last_iter
        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)

        if i == 0:
            logger.info(f"first data batch time: {t_data:.3f}s")
            logger.info(f"LR milestones: {scheduler.milestones} steps.")

        if conf.quant and "vis_iter" in conf and i % conf.vis_iter == 0 and rank == 0:
            model.vis(step)
            if teacher_model is not None:
                teacher_model.vis(step)

        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(img)

        logits = model(img)

        if teacher_model is not None:
            loss = criterion(logits, label, teacher_logits)
        else:
            loss = criterion(logits, label)

        model.zero_grad()
        loss.backward()
        if conf.dist:
            link.synchronize()
        optimizer.step()

        train_acc.set(accuracy(logits, label, world_size))
        t_iter.set(time() - t0)

        if i % conf.log_iter == 0:
            lr = optimizer.param_groups[0]["lr"]
            eta = get_eta(step, len(train_loader), t_iter.avg())
            logger.info(f"[Step {i:6d} / {len(train_loader):6d}]: LR={lr:.5f}, "
                        f"train_accuracy={train_acc.avg():.3f}%, "
                        f"iter_time={t_iter.avg():.3f}s, "
                        f"data_time={t_data:.3f}, ETA={eta}")
            if tb_logger is not None:
                tb_logger.add_scalar("train/loss", loss.item(), step)
                tb_logger.add_scalar("train/accuracy", train_acc.avg(), step)
                tb_logger.add_scalar("train/learning_rate", lr, step)
            barrier()

        if i % conf.eval_iter == 0:
            eval_acc = evaluate(model, val_loader, step)
            logger.info(f"[Step {i:6d}]: val_accuracy={eval_acc:.3f}%")
            if tb_logger is not None:
                tb_logger.add_scalar("evaluate/accuracy", eval_acc, scheduler.last_iter)
            model.train()
            if eval_acc > best_accuracy or best_model is None:
                best_accuracy = eval_acc
                best_model = {
                    "model": map_to_cpu(model.state_dict()),
                    "opt": map_to_cpu(optimizer.state_dict()),
                    "scheduler": map_to_cpu(scheduler.state_dict()),
                    "accuracy": eval_acc,
                }
            barrier()

        if i % conf.save_iter == 0 and best_model is not None:
            if rank == 0:
                os.makedirs(conf.checkpoint_dir, exist_ok=True)
                save_path = os.path.join(conf.checkpoint_dir, f"checkpoint_best.pth")
                with open(save_path, "wb") as f:
                    torch.save(best_model, f)
            barrier()

        t0 = time()

    if rank == 0:
        save_path = os.path.join(conf.checkpoint_dir, f"checkpoint_final_best.pth")
        with open(save_path, "wb") as f:
            torch.save(best_model, f)
    barrier()
    logger.info(f"Training done at step {scheduler.last_iter}, with best accuracy {best_accuracy:.3f}%.")


def evaluate(model, loader, step):
    model.eval()
    acc = 0.

    with torch.no_grad():
        for i, (img, label) in enumerate(loader):
            if "eval_vis" in conf and conf.eval_vis and i % 10 == 0:
                model.vis(step * len(loader) + i)

            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            logitis = model(img)
            acc += accuracy(logitis, label, world_size)

    return acc / len(loader)


if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn")

    main()

    if conf.dist:
        link.finalize()
