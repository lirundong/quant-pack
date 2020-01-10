# -*- coding: utf-8 -*-

import itertools
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from terminaltables import GithubFlavoredMarkdownTable
from tqdm import tqdm


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
def relative_error(x, x_ref, eps=1e-6):
    x_error = (x - x_ref) / x_ref
    return torch.where(x_ref.abs() > eps, x_error, torch.zeros_like(x_ref))


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
        .expand(*target_shape).contiguous()
    we_unfold = we_unfold.reshape(1, num_channels, 1, num_receptive) \
        .expand(*target_shape).contiguous()
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


def unpack_indices(indices):
    # indices: [N, z], where z the dimension of original tensor
    return tuple(idx.squeeze(1) for idx in indices.split(1, 1))


def get_topk_conditional_indices(x, y, x_cond, y_cond, k=10, sort_by="y"):
    valid_idx = torch.nonzero(x_cond(x) & y_cond(y))
    valid_idx = unpack_indices(valid_idx)  # equivalent to `as_tuple=True` in PyTorch-1.3
    if sort_by == "y":
        abnormal_values = y[valid_idx]
    else:
        abnormal_values = x[valid_idx]
    k = min(k, abnormal_values.numel())
    if k == 0:
        return None, None, None
    else:
        _, topk_idx = torch.topk(abnormal_values, k=k)
        valid_topk_idx = tuple(idx[topk_idx] for idx in valid_idx)
        topk_x = x[valid_idx][topk_idx]
        topk_y = y[valid_idx][topk_idx]
        return valid_topk_idx, topk_x, topk_y


def unfold_conv2d_inputs(*inputs, **conv_params):
    input_shape = inputs[0].shape
    ret = []
    for input in inputs:
        assert input.shape == input_shape
        ret.append(F.unfold(input, **conv_params))
    return ret


def unfold_conv2d_weights(*weights):
    c_out = weights[0].size(0)
    w_shape = weights[0].shape
    ret = []
    for weight in weights:
        assert weight.shape == w_shape
        ret.append(weight.reshape(c_out, -1))
    return ret


def select_conv2d_inputs(n_idx, spatial_idx, *inputs):
    ret = []
    for input in inputs:
        assert input.dim() == 3, "please unfold inputs before selection"
        selected = input[n_idx, :, spatial_idx]
        ret.append(selected)
    return ret


def select_fc_inputs(n_idx, *inputs):
    ret = []
    for input in inputs:
        ret.append(input[n_idx, ...])
    return ret


def select_weights(c_out_idx, *weights):
    ret = []
    for weight in weights:
        ret.append(weight[c_out_idx, ...])
    return ret


def select_topk_by_reference(k, ref, *srcs, dim=-1):
    # input shape: [prev_k, target_dim]
    _, topk_idx = torch.topk(ref, k, dim=dim)  # topk_idx: [prev_k, k]
    ret = []
    for src in srcs:
        src_topk = torch.gather(src, dim, topk_idx)
        ret.append(src_topk)
    return ret


def fetch_inputs_weights_from_output_indices(output_index, output_width, input_err, weight_err, input_fp, weight_fp,
                                             input_q, weight_q, module_type, k=3, **op_params):
    assert module_type in ("Conv2d", "Linear")
    if module_type == "Conv2d":
        # output_idx: [prev_k, 4]
        # input: [batch_size, num_receptive (c_in * k_h * k_w), num_output_spatial (H * W)]
        # weight: [c_out, num_receptive]
        input_err, input_fp, input_q = unfold_conv2d_inputs(input_err, input_fp, input_q, **op_params)
        weight_err, weight_fp, weight_q = unfold_conv2d_weights(weight_err, weight_fp, weight_q)
        n_idx, c_out_idx, y, x = output_index  # each index is 1d tensor with length=`k` in `get_abnormal_indices`
        spatial_idx = y * output_width + x
        # selected input/weight size: [prev_k, num_receptive]
        input_err, input_fp, input_q = select_conv2d_inputs(n_idx, spatial_idx, input_err, input_fp, input_q)
    elif module_type == "Linear":
        n_idx, c_out_idx = output_index
        input_err, input_fp, input_q = select_fc_inputs(n_idx, input_err, input_fp, input_q)
    weight_err, weight_fp, weight_q = select_weights(c_out_idx, weight_err, weight_fp, weight_q)
    # select topk receptive elements form input/weights
    input_fp, input_q = select_topk_by_reference(k, input_err, input_fp, input_q)
    weight_fp, weight_q = select_topk_by_reference(k, weight_err, weight_fp, weight_q)
    return input_fp, input_q, weight_fp, weight_q


def format_1d_tensor(x):
    if torch.is_tensor(x):
        assert x.dim() == 1
        x = x.numpy()
    return "(" + ", ".join(f"{v:.5f}" for v in x) + ")"


def format_1d_tensors(*xs):
    ret = []
    for x in xs:
        ret.append(format_1d_tensor(x))
    return ret


def tensor_to_table(**named_values):
    # header = ["input_err_mean", "output_err", "input_fp", "input_q", "weight_fp", "weight_q"]
    header = []
    body = []
    values = []
    n = None
    for name, value in named_values.items():
        assert torch.is_tensor(value)
        if value.dim() > 1:
            topk = value.size(1)
            name += f"@top{topk}"
        header.append(name)
        values.append(value)
        if n is None:
            n = value.size(0)
        else:
            assert value.size(0) == n
    for i in range(n):
        line = []
        for value in values:
            if value[i].numel() == 1:
                line.append(f"{value[i].item():.5f}")
            else:
                line.append(format_1d_tensor(value[i]))
        body.append(line)
    body.insert(0, header)
    table = GithubFlavoredMarkdownTable(body)
    return table.table


def scalar_to_table(fmt="horizontal", **named_values):
    assert fmt in ("horizontal", "vertical")
    header = []
    body = []
    if fmt == "horizontal":
        for name, value in named_values.items():
            header.append(name)
            body.append(f"{value:.5f}")
        table = [header, body]
    else:
        header = ["item", "value"]
        for name, value in named_values.items():
            if not isinstance(value, str):
                value = f"{value:.5f}"
            body.append([name, value])
        body.insert(0, header)
        table = body
    table = GithubFlavoredMarkdownTable(table)
    return table.table


def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


class RelativeErrorPostProcess:

    def __init__(self, apply_to, ce_loss_from,
                 abnormal_x_range=None, abnormal_y_range=None,
                 ideal_x_range=None, ideal_y_range=None,
                 out_err_topn=10, in_err_topk=3):
        assert len(apply_to) == 2, "currently we only support pair-wise comparison"
        # the first registry is reference, second contains outputs with error
        self.apply_to = apply_to
        self.ce_loss_from = ce_loss_from
        # currently we only filter (small x error, large y error) points
        self.out_err_topn = out_err_topn
        self.in_err_topk = in_err_topk
        if abnormal_x_range is not None or abnormal_y_range is not None:
            self.abnormal_x_cond = lambda x: (abnormal_x_range[0] <= x) & (x < abnormal_x_range[1])
            self.abnormal_y_cond = lambda y: (abnormal_y_range[0] <= y) & (y < abnormal_y_range[1])
        else:
            self.abnormal_x_cond = self.abnormal_y_cond = None
        if ideal_x_range is not None or ideal_y_range is not None:
            self.ideal_x_cond = lambda x: (ideal_x_range[0] <= x) & (x < ideal_x_range[1])
            self.ideal_y_cond = lambda y: (ideal_y_range[0] <= y) & (y < ideal_y_range[1])
        else:
            self.ideal_x_cond = self.ideal_y_cond = None

    def after_iter(self, input_reg, outputs):
        ref_reg = input_reg[self.apply_to[0]]
        with_err_reg = input_reg[self.apply_to[1]]
        per_layer_err = OrderedDict()

        logits_q = outputs[self.ce_loss_from].detach()
        labels = outputs["label"]
        ce_loss = F.cross_entropy(logits_q, labels, reduction="none")
        per_layer_err["per_instance_ce_loss"] = ce_loss.to("cpu").clone()

        for k in tqdm(ref_reg, desc=f"{self.__class__.__name__}"):
            output_name = "output" if "output" in ref_reg[k] else "pre_activation"
            ref_input, ref_weight, ref_output = \
                ref_reg[k]["input"], ref_reg[k]["weight"], ref_reg[k][output_name]
            with_err_input, with_err_weight, with_err_output = \
                with_err_reg[k]["input"], with_err_reg[k]["weight"], with_err_reg[k][output_name]
            input_err = relative_error(with_err_input, ref_input)
            output_err = relative_error(with_err_output, ref_output)
            weight_err = relative_error(with_err_weight, ref_weight)
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
                # "weight_error": weight_err,
                # "weight": ref_reg[k]["weight"],
            }

            if self.abnormal_x_cond or self.abnormal_y_cond:
                abnormal_indices, abnormal_input_err_mean, abnormal_output_err = \
                    get_topk_conditional_indices(input_err_mean, output_err,
                                                 self.abnormal_x_cond, self.abnormal_y_cond, self.out_err_topn)
                if abnormal_indices is None:
                    continue
                output_width = output_err.size(3) if module_type == "Conv2d" else None
                abnormal_x_fp, abnormal_x_q, abnormal_w_fp, abnormal_w_q = \
                    fetch_inputs_weights_from_output_indices(abnormal_indices, output_width,
                                                             input_err, weight_err,
                                                             ref_input, ref_weight,
                                                             with_err_input, with_err_weight,
                                                             module_type, self.in_err_topk, **ref_reg[k].get("param", {}))
                abnormal_report = tensor_to_table(
                    input_err_mean=abnormal_input_err_mean,
                    output_err=abnormal_output_err,
                    output_fp=ref_output[abnormal_indices],
                    output_q=with_err_output[abnormal_indices],
                    input_fp=abnormal_x_fp,
                    input_q=abnormal_x_q,
                    weight_fp=abnormal_w_fp,
                    weight_q=abnormal_w_q,
                )
                per_layer_err[k]["abnormal_report"] = abnormal_report

            if self.ideal_x_cond or self.ideal_y_cond:
                ideal_indices, ideal_input_err_mean, ideal_output_err = \
                    get_topk_conditional_indices(input_err_mean, output_err,
                                                 self.ideal_x_cond, self.ideal_y_cond, self.out_err_topn, sort_by="x")
                if ideal_indices is None:
                    continue
                output_width = output_err.size(3) if module_type == "Conv2d" else None
                ideal_x_fp, ideal_x_q, ideal_w_fp, ideal_w_q = \
                    fetch_inputs_weights_from_output_indices(ideal_indices, output_width,
                                                             input_err, weight_err,
                                                             ref_input, ref_weight,
                                                             with_err_input, with_err_weight,
                                                             module_type, self.in_err_topk, **ref_reg[k].get("param", {}))
                ideal_report = tensor_to_table(
                    input_err_mean=ideal_input_err_mean,
                    output_err=ideal_output_err,
                    output_fp=ref_output[ideal_indices],
                    output_q=with_err_output[ideal_indices],
                    input_fp=ideal_x_fp,
                    input_q=ideal_x_q,
                    weight_fp=ideal_w_fp,
                    weight_q=ideal_w_q,
                )
                per_layer_err[k]["ideal_report"] = ideal_report

            x_lb, x_ub, x_delta = with_err_reg[k]["input_qconf"]
            w_lb, w_ub, w_delta = with_err_reg[k]["weight_qconf"]
            qconf_report = scalar_to_table(
                x_lb=x_lb, x_ub=x_ub, x_delta=x_delta,
                w_lb=w_lb, w_ub=w_ub, w_delta=w_delta,
            )
            per_layer_err[k]["qconf_report"] = qconf_report

        return per_layer_err
