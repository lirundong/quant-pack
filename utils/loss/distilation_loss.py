# -*- coding: utf-8 -*-

"""
Adapted from: https://github.com/peterliht/knowledge-distillation-pytorch/blob/master/model/net.py,
original work is distributed under MIT license:

MIT License

Copyright (c) 2018 Haitong Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["KDistLoss"]


def kd_loss(logits, labels, teacher_logits, soft_weight, temperature):
    return F.kl_div(F.log_softmax(logits / temperature, dim=1),
                    F.softmax(teacher_logits / temperature, dim=1)) \
           * (soft_weight * temperature * temperature) \
           + F.cross_entropy(logits, labels) * (1. - soft_weight)


class KDistLoss(nn.Module):

    def __init__(self, soft_weight, temperature):
        super(KDistLoss, self).__init__()
        self.soft_weight = soft_weight
        self.temperature = temperature

    def forward(self, logits, labels, teacher_logits):
        return kd_loss(logits, labels, teacher_logits, self.soft_weight, self.temperature)
