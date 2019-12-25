# -*- coding: utf-8 -*-

import torch


class CosineDistancePostProcess:

    def __init__(self, apply_to):
        assert len(apply_to) == 2, "currently we only support pair-wise comparison"
        self.apply_to = apply_to

    def after_iter(self, input_reg):
        cos_dists = []
        applied_regs = [input_reg[n] for n in self.apply_to]
        for k in applied_regs[0].keys():
            v1 = applied_regs[0][k].view(-1)
            v2 = applied_regs[1][k].view(-1)
            dist = torch.cosine_similarity(v1, v2, dim=-1)
            cos_dists.append((k, dist.item()))
        return cos_dists
