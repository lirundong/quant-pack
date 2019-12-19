# -*- coding: utf-8 -*-

import json
from argparse import ArgumentParser

import quant_pack.apis as qapi


def main():
    parser = ArgumentParser("`quant_pack` CLI for training and evaluating classification models.")
    parser.add_argument("--config", "-c", required=True,
                        help="configuration file path")
    parser.add_argument("--override", "-O", type=json.loads,
                        help="higher priority configuration (in JSON format)")
    parser.add_argument("--distributed", "-d", action="store_true",
                        help="training in distributed environment (SLURM)")
    parser.add_argument("--port", "-p", type=int,
                        help="distributed communication port")
    parser.add_argument("--seed", "-s", type=int, default=19260817,
                        help="manually setup random seed")
    args = parser.parse_args()
    cfg = qapi.build_cfg(args)
    qapi.init_environment(cfg)
    qapi.train_classifier(cfg)


if __name__ == "__main__":
    main()
