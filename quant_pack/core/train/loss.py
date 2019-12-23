# -*- coding: utf-8 -*-

from typing import Tuple

import torch.nn.functional as F
from mmcv.runner import Hook

__all__ = ["CEKLLoss"]


def _kl_distillation_loss(logits, references, temperature):
    references = references.detach()
    return F.kl_div(F.log_softmax(logits / temperature, dim=1),
                    F.softmax(references / temperature, dim=1),
                    reduction="batchmean") * pow(temperature, 2)


class CEKLLoss(Hook):

    def __init__(self,
                 ce_inputs: Tuple[str, str] = ("fp", "label"),
                 kl_inputs: Tuple[str, str] = ("quant", "fp"),
                 ce_loss_weight_name: str = "ce_loss_weight",
                 kl_loss_weight_name: str = "kl_loss_weight",
                 kl_temperature_name: str = "kl_temperature"):
        self.ce_inputs = ce_inputs
        self.kl_inputs = kl_inputs
        self.ce_loss_weight_name = ce_loss_weight_name
        self.kl_loss_weight_name = kl_loss_weight_name
        self.kl_temperature_name = kl_temperature_name

    def after_iter(self, runner):
        assert hasattr(runner, "named_vars")
        ce_inputs = [runner.outputs[n] for n in self.ce_inputs]
        ce_weight = runner.named_vars[self.ce_loss_weight_name]
        ce_loss = F.cross_entropy(*ce_inputs) * ce_weight

        if all(n in runner.outputs for n in self.kl_inputs):
            kl_inputs = [runner.outputs[n] for n in self.kl_inputs]
            kl_weight = runner.named_vars[self.kl_loss_weight_name]
            kl_temperature = runner.named_vars[self.kl_temperature_name]
            kl_loss = _kl_distillation_loss(*kl_inputs, kl_temperature) * kl_weight
        else:
            kl_loss = 0.

        runner.outputs["loss"] = ce_loss + kl_loss
