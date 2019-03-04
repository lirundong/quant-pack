#!/usr/bin/env bash

while getopts :p:qc:x:m: opt
do
    case "$opt" in
    p) PART=$OPTARG ;;
    q) EXTRA="${EXTRA} --quant";;
    c) CONF_NAME=$OPTARG;;
    x) EXTRA="${EXTRA} --extra ${OPTARG}";;
    m) EXTRA="${EXTRA} --comment ${OPTARG}";;
    esac
done

T=`date +%Y%m%d-%H-%M-%S`
ROOT=..

if [[ ! -f $ROOT/configs/${CONF_NAME}.yaml ]]; then
    echo "config \`${CONF_NAME}\` not exist";
    exit -1
else
    CONF_FILE=${ROOT}/configs/${CONF_NAME}.yaml
    echo "evaluating with configure \`$CONF_FILE\`"
fi

source $HOME/.local/bin/env_spring.sh
source activate r0.2.1

export PYTHONPATH=$ROOT:$PYTHONPATH

GLOG_vmodule=MemcachedClient=-1 srun \
   --mpi=pmi2 -p ${PART} -n1 \
   --gres=gpu:1 --ntasks-per-node=1 \
   --job-name=${CONF_NAME} \
   --kill-on-bad-exit=1 \
python $ROOT/tools/train_val_classifier.py \
  --evaluate \
  --conf-path ${CONF_FILE} \
  ${EXTRA} \
  2>&1 | tee test_${CONF_NAME}_${T}.log
