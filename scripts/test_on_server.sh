#!/usr/bin/env bash

while getopts :p:t:c: opt
do
    case "$opt" in
    p) PART=$OPTARG ;;
    t)
        case $OPTARG in
            basic) TASK=basic_conv;;
            conv) TASK=all_conv;;
            trans) TASK=conv_fc;;
            *) echo "unknown task type $OPTARG"; exit -1; ;;
        esac;;
    c) VARIANT=$OPTARG;;
    esac
done

T=`date +%Y%m%d-%H-%M-%S`
ROOT=..
CONF_NAME=${TASK}_${VARIANT}

if [[ ! -f $ROOT/configs/${CONF_NAME}.yaml ]]; then
    echo "config \`${CONF_NAME}\` not exist";
    exit -1
else
    CONF_FILE=${ROOT}/configs/${CONF_NAME}.yaml
    echo "evaluating with configure \`$CONF_FILE\`"
fi

source $HOME/.local/bin/env_spring.sh
source activate r0.2.0

export PYTHONPATH=$ROOT:$PYTHONPATH

GLOG_vmodule=MemcachedClient=-1 srun \
   --mpi=pmi2 -p ${PART} -n1 \
   --gres=gpu:1 --ntasks-per-node=1 \
   --job-name=${CONF_NAME} \
   --kill-on-bad-exit=1 \
python $ROOT/tools/train_val_classifier.py \
  --evaluate \
  --conf-path ${CONF_FILE} \
  2>&1 | tee test_${CONF_NAME}_${T}.log
