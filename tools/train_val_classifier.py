# -*- coding: utf-8 -*-

import os
import logging
import json
from argparse import ArgumentParser
from time import time
from datetime import datetime
from copy import deepcopy
from pprint import pformat
from itertools import chain

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


BEST_ACCURACY = 0.
EXP_DATETIME = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
CONF = None
DEVICE = None
TB_LOGGER = None
RANK = 0
WORLD_SIZE = 1


def main():
    global BEST_ACCURACY, CONF, DEVICE, TB_LOGGER, RANK, WORLD_SIZE

    parser = ArgumentParser(f"Probabilistic quantization neural networks.")
    parser.add_argument("--conf-path", help="path of configuration file")
    parser.add_argument("--evaluate", "-e", action="store_true", help="evaluate trained model")
    parser.add_argument("--extra", "-x", type=json.loads, help="extra configurations in json format")
    parser.add_argument("--comment", "-m", help="comment for each experiment")
    parser.add_argument("--debug", action="store_true", help="logging debug info")
    args = parser.parse_args()

    with open(args.conf_path, "r", encoding="utf-8") as f:
        CONF = yaml.load(f)
        cli_conf = {k: v for k, v in vars(args).items() if k != "extra" and not k.startswith("__")}
        update_config(CONF, cli_conf)
        if args.extra is not None:
            update_config(CONF, args.extra)
        CONF = edict(CONF)

    RANK, WORLD_SIZE = dist_init()
    CONF.dist = WORLD_SIZE > 1
    logger = init_log(CONF.debug, RANK)
    DEVICE = torch.device(f"cuda:{torch.cuda.current_device()}") if torch.cuda.is_available() else torch.device("cpu")

    logger.debug(f"configurations:\n{pformat(CONF)}")
    logger.debug(f"device: {DEVICE}")

    logger.debug(f"building dataset {CONF.dataset.name}...")
    train_set, val_set = get_dataset(CONF.dataset.name, **CONF.dataset.args)
    logger.debug(f"building training loader...")
    train_loader = DataLoader(train_set,
                              sampler=IterationSampler(train_set, rank=RANK, world_size=WORLD_SIZE,
                                                       **CONF.train_samper_conf),
                              **CONF.train_data_conf)
    logger.debug(f"building validation loader...")
    val_loader = DataLoader(val_set,
                            sampler=DistributedSampler(val_set, WORLD_SIZE, RANK) if CONF.dist else None,
                            **CONF.val_data_conf)

    logger.debug(f"building model `{CONF.arch}`...")
    model = backbone.__dict__[CONF.arch](**CONF.arch_conf).to(DEVICE, non_blocking=True)
    logger.debug(f"build model {model.__class__.__name__} done:\n{model}")

    # TODO: better `param_groups` interface?
    weight_group, quant_param_group = model.get_param_group(CONF.weight_conf, CONF.quant_param_conf)
    # TODO: wrap these two optimizers into one, with conferable schedule
    weight_opt = optim.__dict__[CONF.weight_opt](**weight_group)
    quant_param_opt = optim.__dict__[CONF.quant_param_opt](**quant_param_group)
    global_opt = QuantParamWeightOpt(weight_opt, quant_param_opt,
                                     debug=CONF.debug, world_size=WORLD_SIZE, **CONF.global_opt_conf)
    scheduler = IterationScheduler(weight_opt,
                                   dataset_size=len(train_set), world_size=WORLD_SIZE, **CONF.scheduler_conf)

    if CONF.dist:
        logger.debug(f"register all_reduce gradient hooks to model...")
        model = get_dist_module(model)

    if CONF.get("tb_dir") and RANK == 0 and not CONF.evaluate:
        tb_dir = os.path.join(CONF.tb_dir, f"{EXP_DATETIME}_{CONF.comment}")
        logger.debug(f"creating TensorBoard at: {tb_dir}...")
        os.makedirs(tb_dir, exist_ok=True)
        TB_LOGGER = SummaryWriter(tb_dir)

    if CONF.get("resume_path"):
        logger.debug(f"loading checkpoint at: {CONF.resume_path}...")
        checkpoint = torch.load(CONF.resume_path, DEVICE)
        model_dict = checkpoint["model"] if "model" in checkpoint.keys() else checkpoint
        model.load_state_dict(model_dict, strict=False)
        if CONF.get("reset_p"):
            logger.debug(f"resetting probabilistic parameters...")
            model.reset_p()
        if CONF.get("reset_bounds"):
            logger.debug(f"resetting parameter boundaries...")
            model.reset_boundaries()
        if CONF.get("resume_opt"):
            logger.debug(f"recovering optimizer...")
            # weight_opt.load_state_dict(checkpoint["opt"])
            # quant_param_opt.load_state_dict(checkpoint["quant_opt"])
            global_opt.load_state_dict(checkpoint["opt"])
            BEST_ACCURACY = checkpoint["accuracy"]
            scheduler.load_state_dict(checkpoint["scheduler"])
            logger.debug(f"recovered opt at iteration: {scheduler.last_iter}")

    if CONF.distillation:
        logger.debug("building FP teacher model...")
        teacher = deepcopy(model)
        teacher.full_precision()
        teacher.to(DEVICE, non_blocking=True)
        model.register_teacher(teacher)
    else:
        teacher = None

    logger.debug(f"building criterion {CONF.loss}...")
    criterion = get_loss(CONF.loss, **CONF.loss_args)

    if CONF.debug:
        num_params = 0
        opt_conf = []
        for p in chain(weight_opt.param_groups, quant_param_opt.param_groups):
            num_params += len(p["params"])
            opt_conf.append({k: v for k, v in p.items() if k != "params"})
        logger.debug(f"number of parameters: {num_params}")
        logger.debug(f"optimizer conf:\n{pformat(opt_conf)}")

        logger.debug(f"building diagnoser `{CONF.diagnose.diagnoser.type}` with conf: "
                     f"\n{pformat(CONF.diagnose.diagnoser.args)}")
        model = get_diagnoser(CONF.diagnose.diagnoser.type, model,
                              logger=TB_LOGGER, **CONF.diagnose.diagnoser.args)
        debug_tasks = get_tasks(model, CONF.diagnose.tasks)  # TODO: should we preserve these tasks?

    # if CONF.update_bn:
    #     logger.debug(f"updating BN statistics by unlabeled data")
    #     update_bn_stat(model, train_loader)

    if CONF.evaluate:
        assert CONF.resume_path is not None, f"load state_dict before evaluating"
        step = scheduler.last_iter
        logger.info(f"[Step {step}]: evaluating...")
        # TODO: generalize this
        eval_fp_top1, eval_fp_top5, eval_q_top1, eval_q_top5 = evaluate(model, val_loader, step, CONF.enable_eval_quant)
        logger.info(f"[Step {step:6d}]: val_fp_top1={eval_fp_top1:.3f}%")
        logger.info(f"[Step {step:6d}]: val_fp_top5={eval_fp_top5:.3f}%")
        logger.info(f"[Step {step:6d}]: val_q_top1={eval_q_top1:.3f}%")
        logger.info(f"[Step {step:6d}]: val_q_top5={eval_q_top5:.3f}%")
        return

    train(model, criterion, train_loader, val_loader, global_opt, scheduler, teacher)


def train(model, criterion, train_loader, val_loader, global_opt, scheduler, teacher_model=None):

    def barrier():
        if CONF.dist:
            link.barrier()

    global BEST_ACCURACY
    logger = logging.getLogger("global")

    if CONF.quant_enable_step > 0:
        quant_enable_step = CONF.quant_enable_step
    elif CONF.quant_enable_step == 0:  # disable quantization loss
        quant_enable_step = len(train_loader) * 2  # TODO: is this correct?
    else:
        quant_enable_step = scheduler.milestones[CONF.quant_enable_step]
    logger.debug(f"enable quantization loss at step: {quant_enable_step}")

    best_model = None
    t_iter = AverageMeter(20)
    train_fp_top1 = AverageMeter(20)
    train_fp_top5 = AverageMeter(20)
    train_q_top1 = AverageMeter(20)
    train_q_top5 = AverageMeter(20)
    model.train()
    train_loader.sampler.set_last_iter(scheduler.last_iter)

    t0 = time()
    for i, (img, label) in enumerate(train_loader):
        t_data = time() - t0
        scheduler.step()
        step = scheduler.last_iter
        quant_enabled = quant_enable_step <= step
        log_enabled = i % CONF.log_iter == 0
        get_grad_norm = CONF.debug and quant_enabled and log_enabled
        opt = global_opt.enabled_optimizer(step, quant_enabled)
        img = img.to(DEVICE, non_blocking=True)
        label = label.to(DEVICE, non_blocking=True)

        if i == 0:
            logger.debug(f"first data batch time: {t_data:.3f}s")
            logger.debug(f"warmup: {scheduler.base_lrs[0]:.4f} -> {scheduler.target_lrs[0]:.4f} "
                         f"({scheduler.warmup_iters} steps)")
            logger.debug(f"LR milestones: {scheduler.milestones} steps.")
            logger.debug(f"enable quant-loss at step {quant_enable_step}.")

        # if CONF.quant and "vis_iter" in CONF and i % CONF.vis_iter == 0 and RANK == 0:
        #     model.vis(step)
        #     if teacher_model is not None:
        #         teacher_model.vis(step)

        if teacher_model is not None:
            with torch.no_grad():
                teacher_logits = teacher_model(img)

        if CONF.inv_distillation and step == quant_enable_step:
            logger.debug(f"resetting quantization ranges at iteration {scheduler.last_iter}...")
            model.update_weight_quant_param()
            model.update_activation_quant_param(train_loader, CONF.calibration_steps, CONF.calibration_gamma)
            logger.debug(f"evaluating with calibrated quantization ranges...")
            eval_fp_top1, eval_fp_top5, eval_q_top1, eval_q_top5 = evaluate(model, val_loader, step)
            logger.info(f"[Step {step:6d}]: val_fp_top1={eval_fp_top1:.3f}%")
            logger.info(f"[Step {step:6d}]: val_fp_top5={eval_fp_top5:.3f}%")
            logger.info(f"[Step {step:6d}]: val_q_top1={eval_q_top1:.3f}%")
            logger.info(f"[Step {step:6d}]: val_q_top5={eval_q_top5:.3f}%")

        model.update_task_state()  # TODO: wrap this
        logits = model(img, enable_quant=quant_enabled)

        if teacher_model is not None:
            hard_loss, soft_loss = criterion(logits, label, teacher_logits)
            loss = soft_loss + hard_loss
        elif CONF.inv_distillation:
            logits_fp, logits_q = logits
            hard_loss, soft_loss = criterion(logits_fp, logits_q, label)
            # if get_grad_norm:
            #     loss, hard_grad_norm, soft_grad_norm = global_opt.backward(hard_loss, soft_loss, get_grad_norm=True)
            # else:
            #     loss = global_opt.backward(hard_loss, soft_loss)
            loss = global_opt.backward(hard_loss, soft_loss)
        else:
            loss = criterion(logits, label)
        # loss = loss / WORLD_SIZE / CONF.loss_divisor
        # opt.zero_grad()
        # loss.backward()
        # if CONF.dist:
        #     link.synchronize()
        opt.step()
        model.step_done()  # TODO: wrap this

        # TODO: generalize this
        batch_fp_top1, batch_fp_top5 = accuracy(logits_fp, label, WORLD_SIZE, topk=CONF.topk)
        batch_q_top1, batch_q_top5 = accuracy(logits_q, label, WORLD_SIZE, topk=CONF.topk)
        train_fp_top1.set(batch_fp_top1)
        train_fp_top5.set(batch_fp_top5)
        train_q_top1.set(batch_q_top1)
        train_q_top5.set(batch_q_top5)
        t_iter.set(time() - t0)

        if log_enabled:
            lr_w, lr_q = global_opt.get_lr()
            eta = get_eta(step, len(train_loader), t_iter.avg())
            logger.info(f"[Step {i:6d} / {len(train_loader):6d}]: "
                        f"LR={{weight: {lr_w:.4f}, quant_param: {lr_q:.4f}}}, "
                        f"hard_loss={hard_loss.item():.3f}, "
                        f"soft_loss={soft_loss.item():.3f}, "
                        f"fp_top1={train_fp_top1.avg():.3f}%, "
                        f"quant_top1={train_q_top1.avg():.3f}%, "
                        f"iter_time={t_iter.avg():.3f}s, "
                        f"data_time={t_data:.3f}s, ETA={eta}")
            if TB_LOGGER is not None:
                TB_LOGGER.add_scalar("train/loss/all", loss, step)
                TB_LOGGER.add_scalar("train/loss/cross_entropy", hard_loss.item(), step)
                TB_LOGGER.add_scalar("train/loss/soft_KL", soft_loss.item(), step)
                TB_LOGGER.add_scalar("train/fp_top1", train_fp_top1.avg(), step)
                TB_LOGGER.add_scalar("train/fp_top5", train_fp_top5.avg(), step)
                TB_LOGGER.add_scalar("train/quant_top1", train_q_top1.avg(), step)
                TB_LOGGER.add_scalar("train/quant_top5", train_q_top5.avg(), step)
                TB_LOGGER.add_scalar("train/learning_rate/weight", lr_w, step)
                TB_LOGGER.add_scalar("train/learning_rate/quant_param", lr_q, step)
                # if get_grad_norm:
                    # grad_ratio = param_grad_ratio(model)
                    # for k, v in grad_ratio.items():
                    #     TB_LOGGER.add_scalar(f"grad_ratio/{k}", v, step)
                    # for k, p in chain(model.weight_quant_param.named_parameters(),
                    #                   model.activation_quant_param.named_parameters()):
                    #     TB_LOGGER.add_scalar(f"quant_range/{k}", p.data.item(), step)
                    # TB_LOGGER.add_scalar("grad_norm/hard_loss", hard_grad_norm, step)
                    # TB_LOGGER.add_scalar("grad_norm/soft_loss", soft_grad_norm, step)
            barrier()

        if i > 0 and i % CONF.eval_iter == 0:
            logger.debug(f"evaluating at iteration {step}...")
            eval_fp_top1, eval_fp_top5, eval_q_top1, eval_q_top5 = evaluate(model, val_loader, step, quant_enabled)
            logger.info(f"[Step {i:6d}]: val_fp_top1={eval_fp_top1:.3f}%")
            logger.info(f"[Step {i:6d}]: val_q_top1={eval_q_top1:.3f}%")
            if TB_LOGGER is not None:
                TB_LOGGER.add_scalar("evaluate/fp_top1", eval_fp_top1, step)
                TB_LOGGER.add_scalar("evaluate/fp_top5", eval_fp_top5, step)
                TB_LOGGER.add_scalar("evaluate/quant_top1", eval_q_top1, step)
                TB_LOGGER.add_scalar("evaluate/quant_top5", eval_q_top5, step)

            is_best = eval_q_top1 > BEST_ACCURACY if quant_enabled else eval_fp_top1 > BEST_ACCURACY
            if is_best or best_model is None:
                BEST_ACCURACY = eval_q_top1 if quant_enabled else eval_fp_top1
                best_model = {
                    "model": map_to_cpu(model.state_dict()),
                    "opt": map_to_cpu(global_opt.state_dict()),
                    "scheduler": map_to_cpu(scheduler.state_dict()),
                    "accuracy": BEST_ACCURACY,
                }
            barrier()

        if i % CONF.save_iter == 0 and best_model is not None:
            if RANK == 0:
                model_iter = best_model["scheduler"]["last_iter"]
                os.makedirs(CONF.checkpoint_dir, exist_ok=True)
                save_path = os.path.join(CONF.checkpoint_dir, f"ckpt_{EXP_DATETIME}_i{model_iter}.pth")
                with open(save_path, "wb") as f:
                    torch.save(best_model, f)
            barrier()

        t0 = time()

    if RANK == 0:
        os.makedirs(CONF.checkpoint_dir, exist_ok=True)
        save_path = os.path.join(CONF.checkpoint_dir, f"ckpt_{EXP_DATETIME}_final.pth")
        with open(save_path, "wb") as f:
            torch.save(best_model, f)
        logger.info(f"Training done at step {scheduler.last_iter}, "
                    f"with best accuracy {BEST_ACCURACY:.3f}%, "
                    f"checkpoint: {save_path}")
    barrier()


def evaluate(model, loader, step, enable_quant=True):
    model.eval()
    fp_top1 = 0.
    fp_top5 = 0.
    q_top1 = 0.
    q_top5 = 0.

    with torch.no_grad():
        for i, (img, label) in enumerate(loader):

            if "eval_vis" in CONF and CONF.eval_vis and i % 10 == 0:
                model.vis(step * len(loader) + i)

            img = img.to(DEVICE, non_blocking=True)
            label = label.to(DEVICE, non_blocking=True)
            logits = model(img, enable_quant=enable_quant)
            # TODO: generalize this
            logits_fp, logits_q = logits
            batch_fp_top1, batch_fp_top5 = accuracy(logits_fp, label, WORLD_SIZE, topk=CONF.topk)
            batch_q_top1, batch_q_top5 = accuracy(logits_q, label, WORLD_SIZE, topk=CONF.topk)
            fp_top1 += batch_fp_top1
            fp_top5 += batch_fp_top5
            q_top1 += batch_q_top1
            q_top5 += batch_q_top5

    model.train()
    fp_top1 /= len(loader)
    fp_top5 /= len(loader)
    q_top1 /= len(loader)
    q_top5 /= len(loader)

    return fp_top1, fp_top5, q_top1, q_top5


if __name__ == "__main__":
    if mp.get_start_method(allow_none=True) != "forkserver":
        mp.set_start_method("forkserver")

    main()

    if CONF.dist:
        link.finalize()
