# -*- coding: utf-8 -*-

import collections

from torch.optim import Optimizer

__all__ = ["IterationScheduler"]


class IterationScheduler(object):
    def __init__(self, optimizer, milestones, dataset_size, batch_size, world_size=1, gamma=0.1, last_iter=-1):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
        self.optimizer = optimizer
        batch_size *= world_size
        epoch_size = dataset_size // batch_size
        if last_iter == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        if not isinstance(milestones, collections.Iterable):
            milestones = [milestones, ]
        self.milestones = [m * epoch_size for m in milestones]
        self.epoch_size = epoch_size
        self.gamma = gamma
        self.last_iter = last_iter
        self.next_milestone = min([m for m in self.milestones if m >= last_iter])

        self.step(last_iter + 1)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_lr(self):
        return [lr * self.gamma for lr in self.base_lrs]

    def step(self, iteration=None):
        if iteration is None:
            iteration = self.last_iter + 1
        self.last_iter = iteration
        if self.next_milestone is None or self.last_iter < self.next_milestone:
            return

        remain_milestones = [m for m in self.milestones if m > self.last_iter]
        if len(remain_milestones) > 0:
            self.next_milestone = min(remain_milestones)
        else:
            self.next_milestone = None
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
