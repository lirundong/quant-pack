# -*- coding: utf-8 -*-

from collections import OrderedDict

import torch


def cos_dist(v1, v2):
    assert torch.is_tensor(v1) and torch.is_tensor(v2)
    return torch.cosine_similarity(v1.view(-1), v2.view(-1), dim=-1).item()


def norm(x, p="fro"):
    assert torch.is_tensor(x)
    return x.norm(p).item()


class MultiLossGradDist:

    plot_method = "multi_loss_cosine"

    def __init__(self, metric, apply_to):
        if metric == "cosine":
            self.dist_func = cos_dist
        else:
            raise ValueError(f"invalid distance metric: {metric}")
        self.apply_to = apply_to

    def after_iter(self, input_reg, outputs):
        per_layer_grad_dists = OrderedDict()
        for layer_name, dist_dict in input_reg[self.apply_to].items():
            assert len(dist_dict) == 2
            (k1, v1), (k2, v2) = dist_dict.items()
            dist = self.dist_func(v1, v2)
            per_layer_grad_dists[f"{k1}_to_{k2}/{layer_name}"] = dist
            per_layer_grad_dists[f"{k1}_norm/{layer_name}"] = norm(v1)
            per_layer_grad_dists[f"{k2}_norm/{layer_name}"] = norm(v2)
        return per_layer_grad_dists
