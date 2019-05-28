# -*- coding: utf-8 -*-

import os
from collections import defaultdict
from random import shuffle
from argparse import ArgumentParser


def get_imagenet_subset(src, dst, labels, n=100):
    assert os.path.exists(src)
    dst_dir, _ = os.path.split(dst)
    os.makedirs(dst_dir, exist_ok=True)
    meta = defaultdict(list)

    with open(src, "r", encoding="utf-8") as f:
        for l in f.readlines():
            img, label = l.strip().split()
            meta[int(label)].append(img)

    with open(dst, "w", encoding="utf-8") as f:
        for label in labels:
            imgs = meta[int(label)]
            shuffle(imgs)
            for img in imgs[:n]:
                f.write(f"{img} {label}\n")


if __name__ == "__main__":
    parser = ArgumentParser("dataset manipulation tool")
    parser.add_argument("--src", nargs="+", help="paths to source meta files")
    parser.add_argument("--dst", nargs="+", help="paths to destination meta files")
    parser.add_argument("-n", default=1000, help="number of instances in each class")
    parser.add_argument("-c", default=200, help="number of classes")

