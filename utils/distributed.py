import os
from types import MethodType

import torch
import linklink as link

__all__ = ["get_dist_module", "dist_init"]


def get_dist_module(model):

    def _broadcast_params(m):
        for name, p in m.state_dict().items():
            link.broadcast(p.data, 0)

    def _register_hooks(self):
        assert len(self._hook_handles) == 0, "sync_grad hook already resisted"
        for i, (name, p) in enumerate(self.named_parameters()):
            if p.requires_grad:
                # create dummy new leave, such that the gradients w.r.t. p are correctly accumulated
                p_tmp = p.expand_as(p)
                grad_acc = p_tmp.grad_fn.next_functions[0][0]
                h = grad_acc.register_hook(self._make_hook(name, p, i))
                self._grad_accs.append(grad_acc)
                self._hook_handles.append(h)

    def _remove_hooks(self):
        assert len(self._hook_handles) > 0, "sync_grad hook didn't resisted yet"
        for h in self._hook_handles:
            h.remove()
        self._hook_handles.clear()
        self._grad_accs.clear()

    def _make_hook(self, name, p, i):
        def hook(*args):
            link.allreduce_async(name, p.grad.data)
        return hook

    _broadcast_params(model)
    model._grad_accs = []
    model._hook_handles = []
    model._make_hook = MethodType(_make_hook, model)
    model.register_sync_grad_hooks = MethodType(_register_hooks, model)
    model.remove_sync_grad_hooks = MethodType(_remove_hooks, model)
    model.register_sync_grad_hooks()
    return model


def dist_init():
    try:
        proc_id = int(os.environ['SLURM_PROCID'])
        ntasks = int(os.environ['SLURM_NTASKS'])
    except KeyError as e:
        raise RuntimeError(f"this function only works on SenseTime SLURM cluster: {e}")

    if ntasks > 1:
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(proc_id % num_gpus)
        link.initialize()
        world_size = link.get_world_size()
        rank = link.get_rank()
        return rank, world_size
    else:
        return 0, 1
