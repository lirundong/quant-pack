# -*- coding: utf-8 -*-

from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules._functions import SyncBatchNorm as sync_batch_norm

__all__ = ["_reinit_multi_domain"]


def _reinit_multi_domain(primary_module):
    if not primary_module.use_multi_domain:
        return

    def _get_running_stat(module):
        if primary_module.use_multi_domain and primary_module.in_quant_mode:
            running_mean = module.running_mean_q
            running_var = module.running_var_q
            num_batches_tracked = module.num_batches_tracked_q
        else:
            running_mean = module.running_mean
            running_var = module.running_var
            num_batches_tracked = module.num_batches_tracked
        return running_mean, running_var, num_batches_tracked

    def _check_input_dim(x):
        if x.dim() <= 2:
            raise ValueError(f'expected at least 3D input (got {x.dim()}D input)')

    def _multi_domain_bn_forward(module, x):
        _check_input_dim(x)
        running_mean, running_var, num_batches_tracked = module.get_running_stat()

        if module.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = module.momentum

        if module.training and module.track_running_stats:
            num_batches_tracked.add_(1)
            if module.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = module.momentum

        return F.batch_norm(x, running_mean, running_var, module.weight, module.bias,
                            module.training or not module.track_running_stats,
                            exponential_average_factor, module.eps)

    def _multi_domain_sync_bn_forward(module, x):
        if not x.is_cuda:
            raise ValueError('expected x tensor to be on GPU')

        if not module.ddp_gpu_size:
            raise AttributeError('SyncBatchNorm is only supported within torch.nn.parallel.DistributedDataParallel')

        _check_input_dim(x)

        exponential_average_factor = 0.0
        running_mean, running_var, num_batches_tracked = module.get_running_stat()

        if module.training and module.track_running_stats:
            num_batches_tracked.add_(1)
            if module.momentum is None:  # use cumulative moving average
                exponential_average_factor = 1.0 / num_batches_tracked.item()
            else:  # use exponential moving average
                exponential_average_factor = module.momentum

        world_size = 1
        process_group = torch.distributed.group.WORLD
        if module.process_group:
            process_group = module.process_group
        world_size = torch.distributed.get_world_size(process_group)

        # fallback to framework BN when synchronization is not necessary
        if world_size == 1 or (not module.training and module.track_running_stats):
            return F.batch_norm(x, running_mean, running_var, module.weight, module.bias,
                                module.training or not module.track_running_stats,
                                exponential_average_factor, module.eps)
        else:
            return sync_batch_norm.apply(
                x, module.weight, module.bias, running_mean, running_var,
                module.eps, exponential_average_factor, process_group, world_size)

    for m in primary_module.modules():
        if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)):
            assert m._version == 2, f"deprecated batchnorm implementation, " \
                                    f"please update to pytorch>=1.1"
            device = next(m.buffers()).device
            if not hasattr(m, "running_mean_q"):
                m.register_buffer("running_mean_q", torch.zeros(m.num_features, device=device))
            if not hasattr(m, "running_var_q"):
                m.register_buffer("running_var_q", torch.ones(m.num_features, device=device))
            if not hasattr(m, "num_batches_tracked_q"):
                m.register_buffer("num_batches_tracked_q", torch.tensor(0, dtype=torch.long, device=device))

            m.get_running_stat = MethodType(_get_running_stat, m)
            if isinstance(m, nn.BatchNorm2d):
                m.forward = MethodType(_multi_domain_bn_forward, m)
            elif isinstance(m, nn.SyncBatchNorm):
                m.forward = MethodType(_multi_domain_sync_bn_forward, m)
