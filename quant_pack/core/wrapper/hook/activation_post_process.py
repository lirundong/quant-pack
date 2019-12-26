# -*- coding: utf-8 -*-

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


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


# TODO: move following error analysis to separated module
def relative_error(x, x_ref):
    x_error = (x - x_ref) / x_ref
    return torch.where(x_ref > 0, x_error, torch.zeros_like(x_ref))


def conv2d_input_weight_analysis(input_err, weight_err, kernel_size,
                                 dilation=1, padding=0, stride=1):
    # unfold / img2col input -> (N, num_receptive, num_output_spatial_size)
    # unfold weight -> (num_kernels, num_receptive)
    # the output shape is (N, num_kernels, num_output_spatial_size)
    # so we concatenate the `num_receptive` dimension and do mean/std analysis
    ie_unfold = F.unfold(input_err, kernel_size, dilation, padding, stride)
    batch_size, num_receptive, num_output_spatial = ie_unfold.shape
    num_channels = weight_err.size(0)
    we_unfold = weight_err.reshape(num_channels, -1)
    assert we_unfold.size(1) == num_receptive

    target_shape = (batch_size, num_channels, num_output_spatial, num_receptive)
    ie_unfold = ie_unfold.permute(0, 2, 1) \
        .reshape(batch_size, 1, num_output_spatial, num_receptive) \
        .expand(*target_shape)
    we_unfold = we_unfold.reshape(1, num_channels, 1, num_receptive) \
        .expand(*target_shape)
    error_unfold = torch.cat([ie_unfold, we_unfold], dim=-1)
    error_mean = torch.mean(error_unfold, dim=-1)
    error_std = torch.std(error_unfold, dim=-1)
    return error_mean, error_std


def fc_input_weight_analysis(input_err, weight_err):
    batch_size, in_channels = input_err.shape
    out_channels, in_channels = weight_err.shape
    target_shape = (batch_size, out_channels, in_channels)
    input_err = input_err.reshape(batch_size, 1, in_channels) \
        .expand(*target_shape)
    weight_err = weight_err.reshape(1, out_channels, in_channels) \
        .expand(*target_shape)
    error_unfold = torch.cat([input_err, weight_err], dim=-1)
    error_mean = torch.mean(error_unfold, dim=-1)
    error_std = torch.std(error_unfold, dim=-1)
    return error_mean, error_std


class RelativeErrorPostProcess:

    def __init__(self, apply_to):
        assert len(apply_to) == 2, "currently we only support pair-wise comparison"
        # the first registry is reference, second contains outputs with error
        self.apply_to = apply_to

    def after_iter(self, input_reg):
        ref_reg = input_reg[self.apply_to[0]]
        with_err_reg = input_reg[self.apply_to[1]]
        per_layer_err = OrderedDict()
        for k in ref_reg:
            input_err = relative_error(with_err_reg[k]["input"], ref_reg[k]["input"])
            output_err = relative_error(with_err_reg[k]["output"], ref_reg[k]["output"])
            weight_err = relative_error(with_err_reg[k]["weight"], ref_reg[k]["weight"])
            module_type = ref_reg[k]["type"]
            if module_type == "Conv2d":
                conv_param = ref_reg[k]["param"]
                input_err_mean, input_err_std = conv2d_input_weight_analysis(input_err, weight_err, **conv_param)
            elif module_type == "Linear":
                input_err_mean, input_err_std = fc_input_weight_analysis(input_err, weight_err)
            else:
                raise ValueError(f"unsupported type `{module_type}` for {self.__class__.__name__}")
            input_err_mean = input_err_mean.reshape_as(output_err)
            input_err_std = input_err_std.reshape_as(output_err)
            per_layer_err[k] = {
                "input_error_mean": input_err_mean,
                "input_error_std": input_err_std,
                "input_error": input_err,
                "output_error": output_err,
            }
        return per_layer_err
