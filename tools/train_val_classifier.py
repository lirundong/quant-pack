# -*- coding: utf-8 -*-

import os
import logging
import json
from argparse import ArgumentParser
from time import time
from datetime import datetime

import yaml
import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.multiprocessing as mp
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from tensorboardX import SummaryWriter
from easydict import EasyDict as edict

import backbone
from utils import *


best_accuracy = 0.
exp_datetime = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
conf = None
device = None
tb_logger = None


def main():
    global best_accuracy, conf, device, tb_logger

    level = logging.INFO
    logger = logging.getLogger("global")
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(level)
    logger.addHandler(ch)

    parser = ArgumentParser(f"Probabilistic quantization neural networks.")
    parser.add_argument("--conf-path", help="path of configuration file")
    parser.add_argument("--evaluate", "-e", action="store_true", help="evaluate trained model")
    parser.add_argument("--quant", "-q", action="store_true", help="evaluate trained model")
    parser.add_argument("--extra", "-x", type=json.loads, help="extra configurations in json format")
    args = parser.parse_args()

    with open(args.conf_path, "r", encoding="utf-8") as f:
        conf = yaml.load(f)
        conf.update(vars(args))
        if args.extra is not None:
            conf.update(args.extra)
        conf = edict(conf)

    device = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")

    dataset = CIFAR100Sub if "cifar100" in conf and conf.cifar100 else CIFAR10
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_set = dataset(conf.data_dir, train=True, download=True,
                        transform=transforms.Compose([
                            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            normalize,
                        ]))
    train_loader = DataLoader(train_set, sampler=IterationSampler(train_set, **conf.train_samper_conf),
                              **conf.train_data_conf)
    val_set = dataset(conf.data_dir, train=False, download=True,
                      transform=transforms.Compose([
                          transforms.ToTensor(),
                          normalize,
                      ]))
    val_loader = DataLoader(val_set, **conf.val_data_conf)

    model = backbone.__dict__[conf.arch](**conf.arch_conf).to(device, non_blocking=True)
    model.quant(conf.quant)
    criterion = nn.CrossEntropyLoss().to(device, non_blocking=True)
    optimizer = optim.__dict__[conf.opt](model.opt_param_groups(conf.quant, **conf.opt_conf),
                                         **conf.opt_conf)
    scheduler = IterationScheduler(optimizer, dataset_size=len(train_set), **conf.scheduler_conf)

    if "resume_path" in conf:
        checkpoint = torch.load(conf.resume_path, device)
        model.load_state_dict(checkpoint["model"], strict=False)
        if "reset_p" in conf and conf.reset_p:
            model.reset_p()
        if "resume_opt" in conf and conf.resume_opt:
            optimizer.load_state_dict(checkpoint["opt"])
            best_accuracy = checkpoint["accuracy"]
            scheduler.load_state_dict(checkpoint["scheduler"])

    if conf.evaluate:
        assert conf.resume_path is not None, f"load state_dict before evaluating"
        accuracy = evaluate(model, val_loader)
        logger.info(f"[Step {scheduler.last_iter}]: accuracy {accuracy:.3f}%.")
        return

    if "tb_dir" in conf:
        tb_dir = os.path.join(conf.tb_dir, exp_datetime)
        os.makedirs(tb_dir, exist_ok=True)
        tb_logger = SummaryWriter(tb_dir)
        model.register_vis(tb_logger)

    train(model, criterion, train_loader, val_loader, optimizer, scheduler)


def train(model, criterion, train_loader, val_loader, optimizer, scheduler):
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

        if i == 0:
            logger.info(f"first data batch time: {t_data:.3f}s")
            logger.info(f"LR milestones: {scheduler.milestones} steps.")

        if conf.quant and "vis_iter" in conf and i % conf.vis_iter == 0:
            model.vis(step)

        img = img.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        logits = model(img)
        loss = criterion(logits, label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc.set(accuracy(logits, label))
        t_iter.set(time() - t0)

        if i % conf.log_iter == 0:
            lr = optimizer.param_groups[0]["lr"]
            eta = get_eta(step, len(train_loader), t_iter.avg())
            logger.info(f"[Step {step:6d} / {len(train_loader):6d}]: LR={lr:.5f}, "
                        f"train_accuracy={train_acc.avg():.3f}%, "
                        f"iter_time={t_iter.avg():.3f}s, "
                        f"data_time={t_data:.3f}, ETA={eta}")
            if tb_logger is not None:
                tb_logger.add_scalar("train/loss", loss.item(), step)
                tb_logger.add_scalar("train/accuracy", train_acc.avg(), step)
                tb_logger.add_scalar("train/learning_rate", lr, step)

        if i % conf.eval_iter == 0:
            eval_acc = evaluate(model, val_loader)
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

        if i % conf.save_iter == 0 and best_model is not None:
            os.makedirs(conf.checkpoint_dir, exist_ok=True)
            save_path = os.path.join(conf.checkpoint_dir, f"checkpoint_best.pth")
            with open(save_path, "wb") as f:
                torch.save(best_model, f)

        t0 = time()

    save_path = os.path.join(conf.checkpoint_dir, f"checkpoint_final_best.pth")
    with open(save_path, "wb") as f:
        torch.save(best_model, f)
    logger.info(f"Training done at step {scheduler.last_iter}, with best accuracy {best_accuracy:.3f}%.")


def evaluate(model, loader):
    model.eval()
    acc = 0.
    with torch.no_grad():
        for i, (img, label) in enumerate(loader):
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            logitis = model(img)
            acc += accuracy(logitis, label)

    return acc / len(loader)


if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn")

    main()
