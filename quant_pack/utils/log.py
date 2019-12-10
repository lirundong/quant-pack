# -*- coding: utf-8 -*-

import datetime
import time
import logging
import sys
import os
import builtins as _builtins
from collections import defaultdict, deque

import torch
import torch.distributed as dist
from tqdm import tqdm

_rank = 0
__all__ = ["init_log", "MetricLogger", "SmoothedValue"]


def _is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def init_log(name="global", debug=False, log_file=None):
    global _rank
    if _is_dist_avail_and_initialized():
        _rank = dist.get_rank()

    logger = logging.getLogger(name)
    level = logging.DEBUG if debug else logging.INFO
    logger.setLevel(level)
    logger.addFilter(lambda record: _rank == 0)

    log_fmt = "%(asctime)s-%(filename)s#%(lineno)d: [%(levelname)s] %(message)s"
    date_fmt = "%Y-%m-%d_%H:%M:%S"
    fmt = logging.Formatter(log_fmt, date_fmt)

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(level)
    stdout_handler.setFormatter(fmt)
    logger.addHandler(stdout_handler)

    if log_file is not None and _rank == 0:
        log_dir, _ = os.path.split(log_file)
        os.makedirs(log_dir, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None, track_global_stat=False):
        if fmt is None:
            if track_global_stat:
                fmt = "{value:.4f} ({global_avg:.4f})"
            else:
                fmt = "{value:.4f} ({avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt
        self.track_global_stat = track_global_stat

    def update(self, value, n=1):
        if torch.is_tensor(value):
            assert value.numel() == 1
            value = value.item()
        self.deque.append(value)
        if self.track_global_stat:
            self.count += n
            self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not _is_dist_avail_and_initialized() or not self.track_global_stat:
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device="cuda")
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        assert self.track_global_stat
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        fmt_tokens = dict(median=self.median, avg=self.avg, max=self.max, value=self.value)
        if self.track_global_stat:
            fmt_tokens["global_avg"] = self.global_avg
        return self.fmt.format(**fmt_tokens)


class MetricLogger(object):
    def __init__(self, tensorboard=None, delimiter=", ", last_iter=-1, **kwargs):
        self.meters = defaultdict(lambda: SmoothedValue(**kwargs))
        self.tensorboard = tensorboard
        self.delimiter = delimiter
        self.last_iter = last_iter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, (tuple, list)):
                assert len(v) == 2
                self.meters[k].update(*v)
            elif isinstance(v, dict):
                assert len(v) == 2
                self.meters[k].update(**v)
            else:
                self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("`{}` object has no attribute `{}`".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def get_meter(self, *meter_names):
        ret = []
        for name in meter_names:
            val = self.meters[name].global_avg
            ret.append(val)
        return ret

    def log_every(self, iterable, log_freq, log_prefix="", logger_name="global", progress_bar=True):
        MB = 1024.0 * 1024.0
        logger = logging.getLogger(logger_name)
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.4f}")
        data_time = SmoothedValue(fmt="{avg:.4f}")
        space_fmt = f":{len(str(len(iterable)))}d"
        log_tokens = [
            log_prefix,
            "[{0" + space_fmt + "}/{1}]",
            "eta: {eta}",
            "{meters}",
            "time: {time}",
            "data: {data}",
        ]
        if torch.cuda.is_available():
            log_tokens.append("max GRAM: {memory:.0f}MB")
        log_msg = self.delimiter.join(log_tokens)

        if progress_bar:
            _builtin_print = _builtins.print
            _builtins.print = tqdm.write
            if _is_dist_avail_and_initialized():
                rank = dist.get_rank()
                log_prefix = f"[RANK {rank:2d}] {log_prefix}"
            iterable = tqdm(iterable, log_prefix)

        for i, obj in enumerate(iterable):
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)

            step = self.last_iter + i
            if step % log_freq == 0:
                self.synchronize_between_processes()
                eta_seconds = iter_time.avg * (len(iterable) - step)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if progress_bar:
                    print("\n")
                if torch.cuda.is_available():
                    logger.info(log_msg.format(
                        step, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    logger.info(log_msg.format(
                        step, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))

                if self.tensorboard is not None:
                    for name, meter in self.meters.items():
                        if name.startswith("eval_") or name.endswith("_w"):
                            self.tensorboard.add_scalar(f"{log_prefix}/{name}", meter.value, step)
                        else:
                            self.tensorboard.add_scalar(f"{log_prefix}/{name}", meter.avg, step)
            end = time.time()

        if progress_bar:
            _builtins.print = _builtin_print

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info("{} Total time: {}".format(log_prefix, total_time_str))
