# -*- coding: utf-8 -*-

import glob
import os
from argparse import ArgumentParser

import torch
from tqdm import tqdm

CPU = torch.device("cpu")


def main():
    parser = ArgumentParser("reformat pytorch checkpoints to 1.3 CPU format")
    parser.add_argument("--work-dir", default=".")
    parser.add_argument("--output-dir", default="./reformatted_ckpt/")
    parser.add_argument("--inputs", nargs="+", default=glob.glob("./*.pth"))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    for ckpt_name in tqdm(args.inputs):
        ckpt_path = os.path.join(args.work_dir, ckpt_name)
        output_path = os.path.join(args.output_dir, ckpt_name)
        assert os.path.isfile(ckpt_path)
        with open(ckpt_path, "rb") as f:
            ckpt = torch.load(f, CPU)
            torch.save(ckpt, open(output_path, "wb"))


if __name__ == "__main__":
    main()
