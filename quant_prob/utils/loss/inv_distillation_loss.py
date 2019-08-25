# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["InvDistilLoss"]


class InvDistilLoss(nn.Module):

    def __init__(self, soft_weight, temperature, soft_loss_type="KL"):
        assert soft_loss_type in ("KL", "L2", "cos")
        if soft_weight is not None:
            assert 0. <= soft_weight <= 1.
        super(InvDistilLoss, self).__init__()
        self.soft_weight = soft_weight
        self.temperature = temperature
        self.soft_loss_type = soft_loss_type

    def forward(self, logits_fp, logits_q, labels, logits_ref=None):
        if logits_fp is not None:
            hard_loss = F.cross_entropy(logits_fp, labels)
        else:
            hard_loss = torch.tensor(0., device=labels.device)

        if logits_q is not None:
            if self.soft_loss_type == "KL":
                # manually batch_mean
                batch_size = logits_q.size(0)
                soft_loss = F.kl_div(F.log_softmax(logits_q / self.temperature, dim=1),
                                     F.softmax(logits_fp / self.temperature, dim=1),
                                     reduction="sum") \
                            / batch_size * pow(self.temperature, 2)
            elif self.soft_loss_type == "L2":
                soft_loss = F.mse_loss(F.log_softmax(logits_fp / self.temperature),
                                       F.log_softmax(logits_q / self.temperature)) \
                            * pow(self.temperature, 2)
            elif self.soft_loss_type == "cos":
                cos_label = torch.ones(logits_fp.size(0), device=logits_fp.device)
                soft_loss = F.cosine_embedding_loss(logits_fp, logits_q, cos_label)
            else:
                raise ValueError(f"unknown soft loss type: {self.soft_loss_type}")

            if self.soft_weight is not None:
                soft_loss = soft_loss * self.soft_weight
                hard_loss = hard_loss * (1. - self.soft_weight)
        else:
            soft_loss = torch.tensor(0., device=labels.device)

        if logits_ref is not None:
            assert logits_fp is not None
            assert not logits_ref.requires_grad
            batch_size = logits_fp.size(0)
            ref_loss = F.kl_div(F.log_softmax(logits_fp / self.temperature, dim=1),
                                F.softmax(logits_ref / self.temperature, dim=1),
                                reduction="sum") \
                        / batch_size * pow(self.temperature, 2)
        else:
            ref_loss = torch.tensor(0., device=labels.device)

        return hard_loss, soft_loss, ref_loss
