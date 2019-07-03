# -*- coding: utf-8 -*-

import collections


__all__ = ["IterationScheduler"]


class IterationScheduler(object):
    def __init__(self, optimizer, milestones, dataset_size, batch_size,
                 warmup_epochs=0, warmup_lr=0, world_size=1, gamma=0.1,
                 last_iter=-1, enable_quant_at=None, verbose=False):
        batch_size *= world_size
        iters_per_epoch = dataset_size // batch_size
        if last_iter == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))

        if warmup_lr > 0:
            self.base_lrs = [warmup_lr, ] * len(optimizer.param_groups)
        else:
            self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.target_lrs = list(map(lambda group: group['initial_lr'] * world_size, optimizer.param_groups))

        if not isinstance(milestones, collections.Iterable):
            milestones = [milestones, ]

        self.milestones = [m * iters_per_epoch for m in milestones]
        self.warmup_iters = warmup_epochs * iters_per_epoch
        self.iters_per_epoch = iters_per_epoch
        self.world_size = world_size
        self.gamma = gamma
        self.optimizer = optimizer
        self.last_iter = last_iter
        self.in_warmup = self.last_iter < self.warmup_iters

        if enable_quant_at == "begin":
            self.enable_quant_at = 1  # step starts from 1
        elif enable_quant_at is not None:
            self.enable_quant_at = self.milestones[enable_quant_at]
        else:
            self.enable_quant_at = None

        self._update_next_milestone()
        self.step(last_iter + 1)

        if verbose:
            from logging import getLogger
            logger = getLogger("global")
            logger.info(str(self))

    def _update_next_milestone(self):
        remain_milestones = [m for m in self.milestones if m > self.last_iter]
        self.next_milestone = min(remain_milestones) if len(remain_milestones) > 0 else None

    def __str__(self):
        info_tokens = [f"milestone iters: {self.milestones}", ]
        if self.warmup_iters > 0:
            info_tokens += [f"warmup iters: {self.warmup_iters}",
                            f"LR before warmup: {self.base_lrs}",
                            f"LR after warmup: {self.target_lrs}"]
        if self.enable_quant_at is not None:
            info_tokens += [f"enable quantization at iter {self.enable_quant_at}"]

        return "\n".join(info_tokens)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def step(self, iteration=None):
        if iteration is None:
            iteration = self.last_iter + 1
        self.last_iter = iteration

        # linear warmup
        if self.last_iter < self.warmup_iters:
            self.in_warmup = True
            for param, base_lr, target_lr in zip(self.optimizer.param_groups, self.base_lrs, self.target_lrs):
                lr_delta = (target_lr - base_lr) / self.warmup_iters
                lr = base_lr + lr_delta * self.last_iter
                param["lr"] = lr
            return

        # scale LR by world_size
        if self.last_iter == self.warmup_iters and self.world_size > 1:
            self.in_warmup = False
            for param_group, target_lr in zip(self.optimizer.param_groups, self.target_lrs):
                param_group["lr"] = target_lr
            return

        # LR decay
        if self.next_milestone is None or self.last_iter < self.next_milestone:
            return

        self._update_next_milestone()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.gamma

    @property
    def quant_enabled(self):
        if self.enable_quant_at is not None and self.last_iter >= self.enable_quant_at:
            return True
        else:
            return False

    @property
    def do_calibration(self):
        if self.enable_quant_at is not None and self.last_iter == self.enable_quant_at:
            return True
        else:
            return False

