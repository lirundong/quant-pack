#!/usr/bin/env zsh
set -x

if ! [[ $1 =~ .*.yaml ]]; then
  CONF="${1}.yaml"
else
  CONF=$1
fi
PARTITION=${2}
if (( ${+3} )); then
  NTASKS=${3}
else
  NTASKS=8
fi
if (( ${+4} )); then
  NODES="--nodelist=${4}"
else
  NODES=""
fi

if [ $NTASKS -gt 8 ]; then
  NGPUS=8
else
  NGPUS=$NTASKS
fi
if [ $NTASKS -gt 1 ]; then
  DISTFLAG="--distributed"
else
  DISTFLAG=""
fi
ROOT=$PWD/..
export PYTHONPATH=$ROOT:$PYTHONPATH

GLOG_vmodule=MemcachedClient=-1 srun \
  -p ${PARTITION} \
  -n${NTASKS} \
  ${NODES} \
  --gres=gpu:${NGPUS} \
  --ntasks-per-node=${NGPUS} \
  --job-name=quant_pack \
  --mpi=pmi2 \
  --kill-on-bad-exit=1 \
python $ROOT/tools/train_val_classifier.py \
  --config $ROOT/configs/${CONF} \
  --port $(shuf -i 12000-20000 -n 1) \
  ${DISTFLAG}
