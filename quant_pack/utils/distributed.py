# -*- coding: utf-8 -*-

import os
import socket
import re
from datetime import timedelta

import netifaces
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

__all__ = ["get_devices", "get_ddp_model", "dist_init", "barrier"]


def _get_master_ip_slurm():
    node_list = os.environ["SLURM_JOB_NODELIST"]
    tokens = [token for token in re.split(r"[-,\[\]]", node_list) if token.isdigit()]
    master_ip = ".".join(tokens[:4])
    return master_ip


def _get_ib_interface():
    ibs = [i for i in netifaces.interfaces() if i.startswith("ib")]
    return ibs[0]


def _disable_non_master_print(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def _print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = _print


def dist_init(port, gpu_per_model=1, allow_solo=False):
    try:
        proc_id = int(os.environ["SLURM_PROCID"])
        n_tasks = int(os.environ["SLURM_NTASKS"])
    except KeyError as e:
        if allow_solo:
            return 0, 1
        else:
            raise RuntimeError(f"distributed initialization failed: {e}")

    if n_tasks > 1:
        assert port >= 2048, f"port {port} is reserved"

        master_ip = _get_master_ip_slurm()
        os.environ["MASTER_PORT"] = str(port)
        os.environ["MASTER_ADDR"] = str(master_ip)
        os.environ["WORLD_SIZE"] = str(n_tasks)
        os.environ["RANK"] = str(proc_id)

        gpus = get_devices(gpu_per_model, proc_id)
        if not isinstance(gpus, tuple):
            gpus = (gpus, )
        torch.cuda.set_device(gpus[0])

        if gpu_per_model == 1:
            backend = dist.Backend.NCCL
            os.environ["NCCL_SOCKET_IFNAME"] = _get_ib_interface()
        else:
            backend = dist.Backend.GLOO
            os.environ["GLOO_SOCKET_IFNAME"] = _get_ib_interface()
        dist.init_process_group(backend=backend, timeout=timedelta(seconds=30))

        prefix = f"RANK {proc_id:2d} [{socket.gethostname()}, GPU{tuple(g.index for g in gpus)}]"
        print(f"{prefix}: {backend} master://{master_ip}:{port}", flush=True)

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        _disable_non_master_print(rank == 0)
        dist.barrier()

        return rank, world_size
    else:
        return 0, 1


def get_devices(gpu_per_model, rank=None):
    # if each of the tasks requires 2 GPUs, then assign GPUs by
    # (GPU0, GPU4) -> task0, (GPU1, GPU5) -> task1, ...
    # since NCCL utilize a RING topology
    if rank is None:
        rank = dist.get_rank()
    if torch.cuda.is_available():
        tasks_per_node = torch.cuda.device_count() // gpu_per_model
        fp_device_id = rank % tasks_per_node
        fp_device = torch.device(f"cuda:{fp_device_id}")
        q_device = fp_device if gpu_per_model == 1 else torch.device(f"cuda:{fp_device_id + tasks_per_node}")
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


def barrier():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
    else:
        return
