#!/usr/bin/env zsh
set -xe

ROOT=$PWD/..
export PYTHONPATH=$ROOT:$PYTHONPATH

for (( i = 5; i >= 2; i-- )); do
  CONF=$ROOT/configs/GQ_Nets/resnet20_cifar10_recursive_init/w${i}a${i}.yaml
  python $ROOT/tools/train_val_classifier.py --config ${CONF}
done
