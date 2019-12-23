#!/usr/bin/env bash
set -e

ROOT=$PWD/..
export PYTHONPATH=$ROOT:$PYTHONPATH

if [ -z ${3+x} ]; then
  NODES=""
else
  NODES="--nodelist=${3}"
fi

GLOG_vmodule=MemcachedClient=-1 srun \
  -p ${1:-Test} \
  -n${2:-8} \
  ${NODES} \
  --gres=gpu:8 \
  --ntasks-per-node=8 \
  --job-name=quant_pack \
  --mpi=pmi2 \
  --kill-on-bad-exit=1 \
python $ROOT/tools/train_val_classifier.py \
  --config $ROOT/configs/GQ_Nets/resnet18_base.yaml \
  --port $(shuf -i 12000-20000 -n 1) \
  --distributed \
  --eval-only
