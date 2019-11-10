# -*- coding: utf-8 -*-

import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

__all__ = ["Cutout", "CIFAR100Sub"]


class Cutout:

    def __init__(self, mask_size, p, cutout_inside, mask_color=(0, 0, 0)):
        """Adapted from: https://github.com/hysts/pytorch_cutout"""
        self.mask_size = mask_size
        self.p = p
        self.cutout_inside = cutout_inside
        self.mask_size_half = mask_size // 2
        self.offset = 1 if mask_size % 2 == 0 else 0

        mask_color = torch.tensor(mask_color, dtype=torch.float32)
        mask_color = mask_color.reshape(mask_color.size(0), 1, 1)  # so it can broadcast
        self.mask_color = mask_color

    def __call__(self, image):
        assert torch.is_tensor(image)
        assert image.size(0) == self.mask_color.size(0)

        if np.random.random() > self.p:
            return image

        h, w = image.shape[:2]

        if self.cutout_inside:
            cxmin, cxmax = self.mask_size_half, w + self.offset - self.mask_size_half
            cymin, cymax = self.mask_size_half, h + self.offset - self.mask_size_half
        else:
            cxmin, cxmax = 0, w + self.offset
            cymin, cymax = 0, h + self.offset

        cx = np.random.randint(cxmin, cxmax)
        cy = np.random.randint(cymin, cymax)
        xmin = cx - self.mask_size_half
        ymin = cy - self.mask_size_half
        xmax = xmin + self.mask_size
        ymax = ymin + self.mask_size
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)

        image[:, ymin:ymax, xmin:xmax] = self.mask_color
        return image

    def __repr__(self):
        return self.__class__.__name__ + f": p={self.p}, cutout_inside={self.cutout_inside}"


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
