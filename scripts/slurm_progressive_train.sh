#!/usr/bin/env zsh
set -xe

CONF_DIR=${1}
PARTITION=${2}
if (( ${+3} )); then
  NODES="--nodelist=${3}"
else
  NODES=""
fi

ROOT=$PWD/..
export PYTHONPATH=$ROOT:$PYTHONPATH

for (( i = 5; i >= 2; i-- )); do
  CONF=$ROOT/$CONF_DIR/w${i}a${i}.yaml
  GLOG_vmodule=MemcachedClient=-1 srun \
    -p ${PARTITION} \
    -n1\
    ${NODES} \
    --gres=gpu:1 \
    --ntasks-per-node=1 \
    --job-name=quant_pack \
    --mpi=pmi2 \
    --kill-on-bad-exit=1 \
  python $ROOT/tools/train_val_classifier.py \
    --config ${CONF}
done
