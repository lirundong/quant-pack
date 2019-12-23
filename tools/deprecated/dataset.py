# -*- coding: utf-8 -*-

import os
from collections import defaultdict
from random import shuffle
from argparse import ArgumentParser


def get_imagenet_subset(src, dst, labels, label_map, n=100):
    assert os.path.exists(src)
    dst_dir, _ = os.path.split(dst)
    os.makedirs(dst_dir, exist_ok=True)
    meta = defaultdict(list)

    with open(src, "r", encoding="utf-8") as f:
        print(f"loading csrc meta: `{src}`..")
        for l in f.readlines():
            img, label = l.strip().split()
            meta[int(label)].append(img)

    with open(dst, "w", encoding="utf-8") as f:
        print(f"building subset meta: `{dst}`...")
        for label in labels:
            new_label = label_map[int(label)]
            imgs = meta[int(label)]
            shuffle(imgs)
            for img in imgs[:n]:
                f.write(f"{img} {new_label}\n")


if __name__ == "__main__":
    parser = ArgumentParser("dataset manipulation tool")
    parser.add_argument("--csrc", nargs="+", help="paths to source meta files")
    parser.add_argument("--dst", nargs="+", help="paths to destination meta files")
    parser.add_argument("-n", nargs="+", type=int, help="number of instances in each class")
    parser.add_argument("-c", default=200, type=int, help="number of classes")
    args = parser.parse_args()

    labels = list(range(1000))
    shuffle(labels)
    sub_labels = labels[:args.c]

    # pytorch CrossEntropy asserts labels are in [0, num_classes), so we have
    # to map selected label numbers from [0, 1000) to [0, args.c)
    label_map = dict()
    for new_label, origin_label in enumerate(sub_labels):
        label_map[origin_label] = new_label

    for src, dst, n in zip(args.src, args.dst, args.n):
        get_imagenet_subset(src, dst, sub_labels, label_map, n)
