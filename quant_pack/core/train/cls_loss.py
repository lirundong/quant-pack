# -*- coding: utf-8 -*-

from typing import Tuple
from collections import OrderedDict

import torch
import torch.nn.functional as F
from mmcv.runner import Hook

from .utils import get_scalar


def _kl_distillation_loss(logits, references, temperature, detach):
    if detach:
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
                 kl_temperature_name: str = "kl_temperature",
                 detach_kl_ref: bool = True):
        self.ce_inputs = ce_inputs
        self.kl_inputs = kl_inputs
        self.ce_loss_weight_name = ce_loss_weight_name
        self.kl_loss_weight_name = kl_loss_weight_name
        self.kl_temperature_name = kl_temperature_name
        self.detach_kl_ref = detach_kl_ref

    def _ce_loss(self, runner):
        ce_inputs = [runner.outputs[n] for n in self.ce_inputs]
        ce_weight = runner.named_vars[self.ce_loss_weight_name]
        ce_loss = F.cross_entropy(*ce_inputs)
        return ce_loss, ce_weight

    def _kl_loss(self, runner):
        if all(n in runner.outputs for n in self.kl_inputs):
            kl_inputs = [runner.outputs[n] for n in self.kl_inputs]
            kl_weight = runner.named_vars[self.kl_loss_weight_name]
            kl_temperature = runner.named_vars[self.kl_temperature_name]
            kl_loss = _kl_distillation_loss(*kl_inputs, kl_temperature, self.detach_kl_ref)
        else:
            kl_loss = kl_weight = 0.
        return kl_loss, kl_weight

    def _get_loss(self, runner):
        assert hasattr(runner, "named_vars")
        ce_loss, ce_weight = self._ce_loss(runner)
        kl_loss, kl_weight = self._kl_loss(runner)
        loss = OrderedDict(
            ce_loss=ce_loss * ce_weight,
            kl_loss=kl_loss * kl_weight,
        )
        log_vars = OrderedDict(
            ce_loss=get_scalar(ce_loss),
            kl_loss=get_scalar(kl_loss),
            ce_weight=get_scalar(ce_weight),
            kl_weight=get_scalar(kl_weight),
        )
        return loss, log_vars

    def after_iter(self, runner):
        loss, log_vars = self._get_loss(runner)
        runner.outputs.update(loss)
        runner.log_buffer.update(log_vars)


class CEKLCosineLoss(CEKLLoss):

    def __init__(self,
                 ce_inputs: Tuple[str, str] = ("fp", "label"),
                 kl_inputs: Tuple[str, str] = ("quant", "fp"),
                 cosine_inputs: Tuple[str, str] = ("x1_fp", "x1_quant"),
                 ce_loss_weight_name: str = "ce_loss_weight",
                 kl_loss_weight_name: str = "kl_loss_weight",
                 cosine_loss_weight_name: str = "cosine_loss_weight",
                 kl_temperature_name: str = "kl_temperature"):
        super(CEKLCosineLoss, self).__init__(ce_inputs, kl_inputs,
                                             ce_loss_weight_name, kl_loss_weight_name,
                                             kl_temperature_name)
        self.cosine_inputs = cosine_inputs
        self.cosine_loss_weight_name = cosine_loss_weight_name

    def _cosine_loss(self, runner):
        if all(n in runner.outputs for n in self.cosine_inputs):
            cosine_inputs = [runner.outputs[n] for n in self.cosine_inputs]
            cosine_weight = runner.named_vars[self.cosine_loss_weight_name]
            cosine_loss = F.cosine_embedding_loss(*cosine_inputs, torch.tensor(1., device=cosine_inputs[0].device))
        else:
            cosine_loss = cosine_weight = 0.
        return cosine_loss, cosine_weight

    def _get_loss(self, runner):
        loss, log_vars = super(CEKLCosineLoss, self)._get_loss(runner)
        cos_loss, cos_weight = self._cosine_loss(runner)
        loss.update(cosine_loss=cos_loss * cos_weight)
        log_vars.update(
            cosine_loss=get_scalar(cos_loss),
            cosine_weight=get_scalar(cos_weight),
        )
        return loss, log_vars
