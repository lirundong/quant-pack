#!/usr/bin/env bash
set -e

source $HOME/.local/bin/cuda-9-gcc-4.9.sh
source activate torch-1.3.1

ROOT=$PWD/..
export PYTHONPATH=$ROOT:$PYTHONPATH

GLOG_vmodule=MemcachedClient=-1 srun \
  -p ${1:-VI_AIC_1080TI} \
  -n${2:-8} \
  --gres=gpu:8 \
  --ntasks-per-node=8 \
  --job-name=quant_pack \
  --mpi=pmi2 \
  --kill-on-bad-exit=1 \
python $ROOT/tools/train_classifier.py \
  --config configs/GQ_Nets/resnet18_base.yaml \
  --port $(shuf -i 12000-20000 -n 1) \
  --distributed
