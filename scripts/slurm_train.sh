#!/usr/bin/env zsh
set -e

ROOT=$PWD/..
export PYTHONPATH=$ROOT:$PYTHONPATH

if ! [[ $1 =~ .*.yaml ]]; then
  CONF="${1}.yaml"
else
  CONF=$1
fi

if (( ${+4} )); then
  NODES="--nodelist=${4}"
else
  NODES=""
fi

GLOG_vmodule=MemcachedClient=-1 srun \
  -p ${2:-VI_AIC_1080TI} \
  -n${3:-8} \
  ${NODES} \
  --gres=gpu:8 \
  --ntasks-per-node=8 \
  --job-name=quant_pack \
  --mpi=pmi2 \
  --kill-on-bad-exit=1 \
python $ROOT/tools/train_val_classifier.py \
  --config $ROOT/configs/GQ_Nets/${CONF} \
  --port $(shuf -i 12000-20000 -n 1) \
  --distributed
