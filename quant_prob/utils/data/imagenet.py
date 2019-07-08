# -*- coding: utf-8 -*-

import warnings
import logging
import os
import io

from PIL import Image
from torch.utils.data import Dataset

try:
    import mc
    _mc_available = True
except ImportError:
    _mc_available = False
    warnings.warn("memcached not available")

__all__ = ["ImageNetDataset"]


def read_img_by_pil(img_path, color, mc_client=None):
    img_format = "RGB" if color else "L"
    if mc_client is not None:
        value = mc.pyvector()
        mc_client.Get(img_path, value)
        value_str = mc.ConvertBuffer(value)
        buff = io.BytesIO(value_str)
        try:
            img = Image.open(buff)
        except OSError:  # memcached error
            img = Image.open(img_path)
    else:
        img = Image.open(img_path)
    return img.convert(img_format)


class ImageNetDataset(Dataset):
    def __init__(self, img_dir, meta_dir, train, color=True, transform=None):
        logger = logging.getLogger("global")
        self.img_dir = img_dir
        self.train = train
        self.transform = transform
        self.color = color
        self.metas = []
        if self.train:
            meta_file = os.path.join(meta_dir, "train.txt")
        else:
            meta_file = os.path.join(meta_dir, "val.txt")
        with open(meta_file, "r", encoding="utf-8") as f:
            logger.debug(f"building dataset from {meta_file}")
            for line in f.readlines():
                path, cls = line.strip().split()
                self.metas.append((path, int(cls)))
        logger.debug("read meta done")
        self.mc_client = None
 
    def __len__(self):
        return len(self.metas)

    def __getitem__(self, idx):
        if self.mc_client is None and _mc_available:
            self.mc_client = mc.MemcachedClient.GetInstance(
                "/mnt/lustre/share/memcached_client/server_list.conf",
                "/mnt/lustre/share/memcached_client/client.conf")
        if self.train:
            filename = os.path.join(self.img_dir, 'train', self.metas[idx][0])
        else:
            filename = os.path.join(self.img_dir, 'val', self.metas[idx][0])
        cls = self.metas[idx][1]
        img = read_img_by_pil(filename, self.color, self.mc_client)

        if self.transform is not None:
            img = self.transform(img)
        return img, cls
