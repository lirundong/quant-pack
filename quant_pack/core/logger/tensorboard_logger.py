# -*- coding: utf-8 -*-

import torch
import mmcv
import numpy as np
import matplotlib.pyplot as plt
from mmcv.runner import TensorboardLoggerHook, master_only


def pair_to_seq(*pairs):
    n_seq = len(pairs[0])
    ret = tuple([] for _ in range(n_seq))
    for pair in pairs:
        for i in range(n_seq):
            ret[i].append(pair[i])
    return ret


def to_np_array(*args):
    ret = []
    cpu = torch.device("cpu")
    for arg in args:
        if torch.is_tensor(arg):
            ret.append(arg.to(cpu).numpy())
        elif isinstance(arg, (list, tuple)):
            ret.append(np.array(arg))
        else:
            raise ValueError(f"can not convert type `{type(arg)}` to numpy array")
    return ret


def plot_xy(x, y, title=None):
    assert len(x) == len(y)
    f = plt.figure(figsize=(4, 3), dpi=300)
    ax = f.gca()
    ax.plot(x, y)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    return f


def plot_y_with_label(y, labels, title=None):
    assert len(y) == len(labels)
    f = plt.figure(figsize=(4, 3), dpi=300)
    ax = f.gca()
    ax.plot(y)
    locs = np.arange(0, len(y))
    plt.xticks(locs, labels, rotation=75)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    return f


def plot_hist_with_log_scale(x, title=None):
    f = plt.figure(figsize=(4, 3), dpi=300)
    ax = f.gca()
    x, = to_np_array(x)
    ax.hist(x.reshape(-1), bins=128)
    ax.set_yscale("log", basey=10)
    if title:
        ax.set_title(title)
    plt.tight_layout()
    return f


def plot_xy_scatter(x, y, xrange=None, yrange=None, xlabel=None, ylabel=None, title=None, logscale=False):
    assert x.numel() == y.numel()
    f = plt.figure(figsize=(3, 3), dpi=300)
    ax = f.gca()
    if xrange:
        m = (xrange[0] < x) & (x < xrange[1])
        x, y = x[m], y[m]
    if yrange:
        m = (yrange[0] < y) & (y < yrange[1])
        x, y = x[m], y[m]
    x, y = to_np_array(x, y)
    ax.scatter(x.reshape(-1), y.reshape(-1), marker="+", linewidths=1)
    if logscale:
        ax.set_xscale("log", basex=10)
        ax.set_yscale("log", basey=10)
    if title:
        ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    plt.tight_layout()
    return f


def plot_3d_hist_of_filters(tb_writer, data, label):
    assert torch.is_tensor(data) and data.dim() > 1
    for i, filter in enumerate(data):
        tb_writer.add_histogram(label, filter.reshape(-1), i)


class EnhancedTBLoggerHook(TensorboardLoggerHook):

    @master_only
    def log(self, runner):
        if "plot" in runner.log_buffer.output:
            plot_buf = runner.log_buffer.output.pop("plot")
            for plot_name, plot_data in plot_buf.items():
                # TODO: add `plot_func` parameter in input buffers
                if isinstance(plot_data, dict):  # layer-wise error analysis
                    for layer_name, err_dict in plot_data.items():
                        input_err_mean = err_dict["input_error_mean"]
                        output_err = err_dict["output_error"]
                        tag = f"input_output_error/{layer_name}"
                        fig_scatter = plot_xy_scatter(input_err_mean, output_err,
                                                      xrange=(-10., 10.), yrange=(-10., 10.),  # TODO: remove this ad-hoc
                                                      xlabel="input_error_mean", ylabel="output_error",
                                                      title=layer_name)
                        self.writer.add_figure(tag, fig_scatter, runner.iter)
                        input_err = err_dict["input_error"]
                        tag = f"input_error_hist/{layer_name}"
                        fig_hist = plot_hist_with_log_scale(input_err, title=layer_name)
                        self.writer.add_figure(tag, fig_hist, runner.iter)

                        reports = [k for k in err_dict.keys() if k.endswith("_report")]
                        for report in reports:
                            text = err_dict[report]
                            tag = f"{report}/{layer_name}"
                            self.writer.add_text(tag, text, runner.iter)

                        if "weight" in err_dict:
                            label = f"weight/{layer_name}"
                            plot_3d_hist_of_filters(self.writer, err_dict["weight"], label)
                        if "weight_error" in err_dict:
                            label = f"weight_error/{layer_name}"
                            plot_3d_hist_of_filters(self.writer, err_dict["weight_error"], label)

                else:  # layer-wise cosine distance
                    x, y = pair_to_seq(*plot_data)
                    if mmcv.is_list_of(x, str):
                        fig = plot_y_with_label(y, labels=x, title=plot_name)
                    else:
                        fig = plot_xy(x, y, title=plot_name)
                    tag = f"{plot_name}/{runner.mode}"
                    self.writer.add_figure(tag, fig, runner.iter)
            plot_buf.clear()

        super(EnhancedTBLoggerHook, self).log(runner)
