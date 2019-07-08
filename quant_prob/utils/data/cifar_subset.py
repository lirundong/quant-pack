# -*- coding: utf-8 -*-

import os
import pickle

import numpy as np
from torch.utils.data import Dataset
from PIL import Image

__all__ = ["CIFAR100Sub"]


class CIFAR100Sub(Dataset):

    def __init__(self, root, train, transform=None, target_transform=None, download=False):
        super(CIFAR100Sub, self).__init__()

        self.root = root
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download  # to compatible with CIFAR100 API

        if self.train:
            cifar_dir = os.path.join(self.root, "train")
        else:
            cifar_dir = os.path.join(self.root, "test")
        with open(cifar_dir, "rb") as f:
            cifar_data = pickle.load(f)
        imgs = np.asarray(cifar_data["data"]).astype(np.uint8)
        labels = cifar_data["fine_labels"]
        assert len(imgs) == len(labels), f"unequal sample numbers"
        self.n = imgs.shape[0]
        imgs = imgs.reshape(self.n, 3, 32, 32).transpose(0, 2, 3, 1)
        self.imgs = imgs
        self.labels = labels

    def __len__(self):
        return self.n

    def __getitem__(self, item):
        img = Image.fromarray(self.imgs[item])
        label = self.labels[item]

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label
