# -*- coding: utf-8 -*-

import os
import subprocess
import sys
from argparse import ArgumentParser
from collections import OrderedDict

import torch

_cleanup_info = \
    "Note that due to `deep_copy` in previous implementations, modules are " \
    "part of the old pickle, so you have to assign the directory and git commit hash of previous codebase to " \
    "`--prev-codebase` and `--prev-commit` when calling this tool with `--clean`."


def main():
    parser = ArgumentParser("`quant-pack` CLI for upgrading checkpoints form previous versions.")
    parser.add_argument("--input", "-i", required=True, help="path of input checkpoint")
    parser.add_argument("--output", "-o", required=True, help="output directory for upgraded checkpoint")
    parser.add_argument("--clean", action="store_true",
                        help="cleanup unnecessary pickles in input checkpoint." + _cleanup_info)
    parser.add_argument("--prev-codebase",
                        help="directory of previous codebase, useful when cleaning checkpoints")
    parser.add_argument("--prev-commit",
                        help="hash of git commit in previous codebase, useful when cleaning checkpoints")
    args = parser.parse_args()

    if args.clean:
        assert args.prev_commit is not None and args.prev_codebase is not None
        print(f"reconstructing previous codebase, wait a minute...")
        ret = subprocess.run(["git", "checkout", "-b", args.prev_commit, args.prev_commit],
                             cwd=args.prev_codebase, capture_output=True, encoding="utf-8")
        ret.check_returncode()
        ret = subprocess.run(["python", "setup.py", "build_ext", "--inplace"],
                             cwd=args.prev_codebase, capture_output=True, encoding="utf-8")
        ret.check_returncode()
        sys.path.insert(0, args.prev_codebase)

    ckpt = torch.load(args.input, torch.device("cpu"))
    print(f"loaded input checkpoint: {args.input}")
    if "model" in ckpt:
        ckpt = ckpt["model"]
    # in `quant-pack` v0.1.1, BN parameters and buffers are automatically moved to related Conv/Linear layers, thus we
    # should separate previous checkpoint to two subsets, used in `pre_trained` (load Conv/BN/Linear parameters) and
    # `resume` (load quantization parameters), respectively
    pre_train = OrderedDict()
    resume = OrderedDict()
    resume["meta"] = {"epoch": 0, "iter": 0}  # dummy meta for adaptation to mmcv
    for k, v in ckpt.items():
        if "activation_quant_param" in k:
            # `activation_quant_param.layer1_0_conv2_act_ub` -> `layer1.0.conv2.a_ub`
            layer_name = k.split(".")[1].replace("_", ".")
            layer_name = layer_name.replace("act.lb", "a_lb", 1)
            layer_name = layer_name.replace("act.ub", "a_ub", 1)
            resume[layer_name] = v.clone()
        elif "weight_quant_param" in k:
            # `weight_quant_param.layer1_1_conv2_weight_lb` -> `layer1.1.conv2.w_lb`
            layer_name = k.split(".")[1].replace("_", ".")
            layer_name = layer_name.replace("weight.lb", "w_lb", 1)
            layer_name = layer_name.replace("weight.ub", "w_ub", 1)
            resume[layer_name] = v.clone()
        else:
            pre_train[k] = v.clone()

    os.makedirs(args.output, exist_ok=True)
    pre_train_ckpt_path = os.path.join(args.output, "pre_trained.pth")
    resume_ckpt_path = os.path.join(args.output, "resume.pth")
    torch.save(pre_train, open(pre_train_ckpt_path, "wb"))
    print(f"upgraded checkpoint for pre_train (#{len(pre_train)} param tensors) written to: {pre_train_ckpt_path}")
    torch.save(resume, open(resume_ckpt_path, "wb"))
    print(f"upgraded checkpoint for resume (#{len(resume)} param tensors) written to: {resume_ckpt_path}")
    print("please use these two files IN CONJUNCTION!")


if __name__ == "__main__":
    main()
