# -*- coding: utf-8 -*-

import os
import re
import socket
from datetime import timedelta
from pathlib import Path

import netifaces
import colorama
import torch
import yaml
import torch.distributed as dist
import torch.multiprocessing as mp
import numpy as np
from easydict import EasyDict
from deepmerge import Merger


def _get_master_ip_slurm():
    node_list = os.environ["SLURM_JOB_NODELIST"]
    tokens = [token for token in re.split(r"[-,\[\]]", node_list) if token.isdigit()]
    master_ip = ".".join(tokens[:4])
    return master_ip


def _get_ib_interface():
    ibs = [i for i in netifaces.interfaces() if i.startswith("ib")]
    return ibs[0]


def _get_device(rank):
    n_gpus = torch.cuda.device_count()
    gpu_id = rank % n_gpus
    return torch.device(f"cuda:{gpu_id}")


def _disable_non_master_print(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def _print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = _print


def _init_dist_and_device(cfg):
    if cfg.distributed:
        try:
            proc_id = int(os.environ["SLURM_PROCID"])
            n_tasks = int(os.environ["SLURM_NTASKS"])
        except KeyError as e:
            raise RuntimeError(f"distributed initialization failed: {e}")
    else:
        proc_id, n_tasks = 0, 1
        cfg.device = torch.device("cuda:0")

    if n_tasks > 1:
        assert cfg.port >= 2048, f"port {cfg.port} is reserved"

        gpu = _get_device(proc_id)
        torch.cuda.set_device(gpu)
        cfg.device = gpu

        master_ip = _get_master_ip_slurm()
        backend = dist.Backend.NCCL
        os.environ["NCCL_SOCKET_IFNAME"] = _get_ib_interface()
        os.environ["MASTER_PORT"] = str(cfg.port)
        os.environ["MASTER_ADDR"] = str(master_ip)
        os.environ["WORLD_SIZE"] = str(n_tasks)
        os.environ["RANK"] = str(proc_id)
        dist.init_process_group(backend=backend, timeout=timedelta(seconds=30))

        prefix = f"RANK {proc_id:2d} [{socket.gethostname()}, {gpu}]"
        print(f"{prefix}: {backend} master://{master_ip}:{cfg.port}", flush=True)

        _disable_non_master_print(proc_id == 0)
        dist.barrier()


def _update_dict_item(src_dict, k, v):
    if "." in k:
        tokens = k.split(".")
        current_k, remain_k = tokens[0], ".".join(tokens[1:])
        src_dict.setdefault(current_k, dict())
        _update_dict_item(src_dict[current_k], remain_k, v)
    else:
        src_dict[k] = v
        return


def build_cfg(args):
    cfg = yaml.load(open(args.config, "r", encoding="utf-8"), Loader=yaml.SafeLoader)

    if cfg["__BASE__"]:
        base_path = cfg.pop("__BASE__")
        if not os.path.exists(base_path):
            # __file__: quant_pack/apis/env.py
            project_root = Path(__file__).absolute().parents[2]
            base_path = os.path.join(project_root, base_path)
        base_cfg = yaml.load(open(base_path, "r", encoding="utf-8"), Loader=yaml.SafeLoader)
    else:
        base_cfg = {}

    # strategies for merging configs:
    #  - override lists in BASE by the counterparts in current config, such that
    #    each param-group can get the latest, non-duplicated config;
    #  - merge dicts recursively, such that derived config can be written as concisely as possible;
    conf_merger = Merger([(list, "override"), (dict, "merge")], ["override"], ["override"])
    conf_merger.merge(base_cfg, cfg)
    cfg = base_cfg

    if args.override is not None:
        for k, v in args.override.items():
            _update_dict_item(cfg, k, v)

    for k, v in vars(args).items():
        if not k.startswith("__") and k != "config" and k != "override":
            cfg[k] = v

    return EasyDict(cfg)


def init_environment(cfg):
    if mp.get_start_method(allow_none=True) != "forkserver":
        mp.set_start_method("forkserver")
    colorama.init()
    _init_dist_and_device(cfg)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
