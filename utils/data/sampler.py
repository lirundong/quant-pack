# -*- coding: utf-8 -*-

import math

import numpy as np
from torch.utils.data import Sampler

__all__ = ["IterationSampler"]


class IterationSampler(Sampler):
    def __init__(self, dataset, batch_size, total_iter=None, total_epoch=None,
                 last_iter=None, last_epoch=None, rank=0, world_size=1, seed=19260817):
        super(IterationSampler, self).__init__(None)

        assert not all(i is not None for i in (total_iter, total_epoch)), \
            f"total_iter and total_epoch can't be both set"
        assert not all(i is not None for i in (last_iter, last_epoch)), \
            f"last_iter and last_epoch can't be both set"

        if total_iter is None:
            total_iter = total_epoch * len(dataset) // batch_size // world_size
        if last_iter is None:
            if last_epoch is not None and last_epoch > 0:
                last_iter = last_epoch * len(dataset) // batch_size // world_size
            else:
                last_iter = -1

        self.dataset = dataset
        self.batch_size = batch_size
        self.total_iter = total_iter
        self.last_iter = last_iter
        self.rank = rank
        self.world_size = world_size
        self.seed = seed

        self.total_size = self.total_iter * self.batch_size
        self.gone_indices = (self.last_iter + 1) * self.batch_size
        self.indices = self.gen_new_list()
        self.call = 0

    def __iter__(self):
        self.call += 1
        indices = self.indices[self.gone_indices:]
        np.random.seed(self.seed + self.call)
        np.random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return len(self.indices)

    def set_last_iter(self, last_iter):
        self.last_iter = last_iter
        self.gone_indices = (self.last_iter + 1) * self.batch_size

    def gen_new_list(self):
        np.random.seed(self.seed)

        dataset_size = len(self.dataset)
        global_size = self.total_size * self.world_size
        indices = np.arange(dataset_size)
        indices = indices[:global_size]
        num_repeat = math.ceil(global_size / dataset_size)
        indices = np.tile(indices, num_repeat)
        indices = indices[:global_size]

        np.random.shuffle(indices)
        begin_idx = self.rank * self.total_size
        indices = indices[begin_idx: begin_idx + self.total_size]
        assert len(indices) == self.total_size

        return indices
