# -*- coding: utf-8 -*-

import math

import numpy as np
from torch.utils.data import Sampler

__all__ = ["IterationSampler"]


class IterationSampler(Sampler):
    def __init__(self, dataset, batch_size, total_iter=None, total_epoch=None,
                 last_iter=None, last_epoch=None, seed=19260817):
        super(IterationSampler, self).__init__(None)

        assert not all(i is not None for i in (total_iter, total_epoch)), \
            f"total_iter and total_epoch can't be both set"
        assert not all(i is not None for i in (last_iter, last_epoch)), \
            f"last_iter and last_epoch can't be both set"

        if total_iter is None:
            total_iter = total_epoch * len(dataset) // batch_size
        if last_iter is None:
            if last_epoch is not None and last_epoch > 0:
                last_iter = last_epoch * len(dataset) // batch_size
            else:
                last_iter = -1

        self.dataset = dataset
        self.batch_size = batch_size
        self.total_iter = total_iter
        self.last_iter = last_iter
        self.seed = seed

        self.total_size = self.total_iter * self.batch_size
        self.gone_indices = (self.last_iter + 1) * self.batch_size
        self.indices = self.gen_new_list()
        self.call = 0

    def __iter__(self):
        if self.call > 0:
            raise RuntimeError("this sampler is not designed to be called more than once!!")

        self.call += 1
        return iter(self.indices[self.gone_indices:])

    def __len__(self):
        # note here we do not take last iter into consideration, since __len__
        # should only be used for displaying, the correct remaining size is
        # handled by DataLoader
        return len(self.indices)

    def set_last_iter(self, last_iter):
        self.last_iter = last_iter
        self.gone_indices = (self.last_iter + 1) * self.batch_size

    def gen_new_list(self):
        # each process shuffle all list with same seed, and pick one piece according to rank
        np.random.seed(self.seed)

        dataset_size = len(self.dataset)
        indices = np.arange(dataset_size)
        indices = indices[:self.total_size]
        num_repeat = math.ceil(self.total_size / dataset_size)
        indices = np.tile(indices, num_repeat)
        indices = indices[:self.total_size]

        np.random.shuffle(indices)
        assert len(indices) == self.total_size

        return indices
