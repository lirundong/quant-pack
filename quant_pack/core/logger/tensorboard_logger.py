# -*- coding: utf-8 -*-

import mmcv
import numpy as np
import matplotlib.pyplot as plt
from mmcv.runner import TensorboardLoggerHook, master_only


def plot_xy(x, y, title=None):
    assert len(x) == len(y)
    f = plt.figure(figsize=(4, 3), dpi=300)
    ax = f.gca()
    ax.plot(x, y)
    if title:
        ax.set_title(title)
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
    return f


def pair_to_seq(*pairs):
    n_seq = len(pairs[0])
    ret = tuple([] for _ in range(n_seq))
    for pair in pairs:
        for i in range(n_seq):
            ret[i].append(pair[i])
    return ret


class EnhancedTBLoggerHook(TensorboardLoggerHook):

    @master_only
    def log(self, runner):
        if "plot" in runner.log_buffer.output:
            plot_buf = runner.log_buffer.output.pop("plot")
            for plot_name, plot_data in plot_buf.items():
                x, y = pair_to_seq(*plot_data)
                if mmcv.is_list_of(x, str):
                    fig = plot_y_with_label(y, labels=x, title=plot_name)
                else:
                    fig = plot_xy(x, y, title=plot_name)
                tag = f"{plot_name}/{runner.mode}"
                self.writer.add_figure(tag, fig, runner.iter)
            plot_buf.clear()

        super(EnhancedTBLoggerHook, self).log(runner)
