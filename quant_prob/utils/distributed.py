# -*- coding: utf-8 -*-

import os
import socket
from time import sleep
from datetime import timedelta

import netifaces
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

__all__ = ["get_devices", "get_ddp_model", "dist_init"]

IP_PATH = "./master_ip"


def _get_master_ip(device="ib0"):  # {eth2, ib0}
    rank = int(os.environ["SLURM_PROCID"])
    if rank == 0:
        master_ip = netifaces.ifaddresses(device)[netifaces.AF_INET][0]["addr"]
        with open(IP_PATH, "w") as f:
            f.write(master_ip)
    else:
        tries = 0
        while not os.path.exists(IP_PATH) and tries < 50:
            sleep(0.1)
            tries += 1
        sleep(0.1)

    with open(IP_PATH, "r") as f:
        master_ip = f.read().strip()

    return master_ip


def _disable_non_master_print(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def _print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = _print


def dist_init(port, gpu_per_model=1, debug=False):
    try:
        proc_id = int(os.environ["SLURM_PROCID"])
        n_tasks = int(os.environ["SLURM_NTASKS"])
    except KeyError as e:
        raise RuntimeError(f"this function only works on SLURM cluster: {e}")

    if n_tasks > 1:
        assert port >= 2048, f"port {port} is reserved"

        global IP_PATH
        IP_PATH = f"{IP_PATH}_{port}"
        master_ip = _get_master_ip()

        os.environ["MASTER_PORT"] = str(port)
        os.environ["MASTER_ADDR"] = str(master_ip)
        os.environ["WORLD_SIZE"] = str(n_tasks)
        os.environ["RANK"] = str(proc_id)
        backend = dist.Backend.NCCL if gpu_per_model == 1 else dist.Backend.GLOO
        dist.init_process_group(backend=backend, timeout=timedelta(seconds=30))

        gpus = get_devices(gpu_per_model)
        if not isinstance(gpus, tuple):
            gpus = (gpus, )
        prefix = f"RANK {proc_id:2d} [{socket.gethostname()}, GPU{tuple(g.index for g in gpus)}]"
        print(f"{prefix}: NCCL master://{master_ip}:{port}", flush=True)

        if debug:
            print(f"{prefix}: testing all_reduce...", flush=True)
            if not isinstance(gpus, (list, tuple)):
                gpus = (gpus, )
                for gpu in gpus:
                    dummy_tensor = torch.randn(24, 128, 128, device=gpu)
                    dist.all_reduce(dummy_tensor)
            print(f"{prefix}: all_reduce done", flush=True)
            dist.barrier()

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        _disable_non_master_print(rank == 0)
        if rank == 0:
            os.remove(IP_PATH)
        dist.barrier()

        return rank, world_size
    else:
        return 0, 1


def get_devices(gpu_per_model):
    # if each of the tasks requires 2 GPUs, then assign GPUs by
    # (GPU0, GPU1) -> task0, (GPU2, GPU3) -> task1, ...
    if torch.cuda.is_available():
        tasks_per_node = torch.cuda.device_count() // gpu_per_model
        fp_device_id = (dist.get_rank() % tasks_per_node) * gpu_per_model
        fp_device = torch.device(f"cuda:{fp_device_id}")
        q_device = fp_device if gpu_per_model == 1 else torch.device(f"cuda:{fp_device_id + 1}")
    else:
        fp_device = q_device = torch.device("cpu")

    if gpu_per_model == 1:
        return fp_device
    else:
        return fp_device, q_device


def get_ddp_model(model, devices=None, debug=False):
    if debug:
        model_devices = {p.device for p in model.parameters()}
        print(f"RANK {dist.get_rank():2d}: model devices={model_devices}", flush=True, force=True)

    model_without_ddp = model
    if devices is None:
        devices = [torch.cuda.current_device(), ]
    elif len(set(devices)) == 1:
        devices = [devices[0], ]
    else:
        devices = None
    ddp_model = DistributedDataParallel(model, device_ids=devices, find_unused_parameters=True)

    return ddp_model, model_without_ddp
