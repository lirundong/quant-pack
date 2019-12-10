# -*- coding: utf-8 -*-
# TODO:
#  1. refactor this module into a diagnosis task

import torch

__all__ = ["get_grad_analysis"]


def get_grad_analysis(model, hard_loss, soft_loss):
    """ Analysis gradients build from unscaled losses.

    Args:
        model:
        hard_loss:
        soft_loss:

    Returns:

    """

    hard_grad_registry = []
    soft_grad_registry = []
    handles = []

    def _get_grad_hook(reg, grad):
        if grad is not None:
            reg.append(grad.detach().clone().reshape(-1))
            return torch.zeros_like(grad)

    def _get_loss_grads(_registry, _loss):
        assert len(handles) == 0
        assert len(_registry) == 0
        for n, p in model.named_parameters():
            if not n.endswith("_quant_param") and p.requires_grad:
                h = p.register_hook(lambda grad: _get_grad_hook(_registry, grad))
                handles.append(h)
        _loss.backward(retain_graph=True)
        for h in handles:
            h.remove()
        handles.clear()
        _grad = torch.cat(_registry)
        return _grad

    hard_grad = _get_loss_grads(hard_grad_registry, hard_loss)
    soft_grad = _get_loss_grads(soft_grad_registry, soft_loss)
    assert hard_grad.numel() == soft_grad.numel()

    n = hard_grad.numel()
    conflict_coeff = (hard_grad.sign() * soft_grad.sign()).sum().item() / n
    hard_grad_norm = hard_grad.norm().item()
    soft_grad_norm = soft_grad.norm().item()

    return conflict_coeff, hard_grad_norm, soft_grad_norm
