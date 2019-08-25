import os
from time import sleep
from datetime import timedelta

import netifaces
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

__all__ = ["get_ddp_model", "dist_init"]

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


def dist_init(port):
    try:
        proc_id = int(os.environ["SLURM_PROCID"])
        n_tasks = int(os.environ["SLURM_NTASKS"])
    except KeyError as e:
        raise RuntimeError(f"this function only works on SLURM cluster: {e}")

    if n_tasks > 1:
        assert port >= 2048, f"port {port} is reserved"

        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(proc_id % num_gpus)

        global IP_PATH
        IP_PATH = f"{IP_PATH}_{port}"
        master_ip = _get_master_ip()

        os.environ["MASTER_PORT"] = str(port)
        os.environ["MASTER_ADDR"] = str(master_ip)
        os.environ["WORLD_SIZE"] = str(n_tasks)
        os.environ["RANK"] = str(proc_id)
        dist.init_process_group(backend="nccl", timeout=timedelta(seconds=5))
        print(f"[RANK {proc_id:2d}@GPU{torch.cuda.current_device()}]: "
              f"NCCL master://{master_ip}:{port}", flush=True)

        rank = dist.get_rank()
        world_size = dist.get_world_size()
        _disable_non_master_print(rank == 0)
        if rank == 0:
            os.remove(IP_PATH)
        dist.barrier()

        return rank, world_size
    else:
        return 0, 1


def get_ddp_model(model):
    device = torch.cuda.current_device()
    model_without_ddp = model
    ddp_model = DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)

    return ddp_model, model_without_ddp
