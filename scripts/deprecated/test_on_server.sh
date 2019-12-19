#!/usr/bin/env bash
set -e

while getopts :p:qc:x:m:n:g: opt
do
    case "$opt" in
    p) PART=$OPTARG ;;
    q) EXTRA="${EXTRA} --quant";;
    c) CONF_NAME=$OPTARG;;
    x) EXTRA="${EXTRA} --extra ${OPTARG}";;
    m) EXTRA="${EXTRA} --comment ${OPTARG}";;
    g) GPUS_PER_JOB=$OPTARG;;
    n) NUM_JOBS=$OPTARG;;
    *) echo "invalid flag -$opt";
       exit 1;;
    esac
done

TASKS_PER_NODE=$[8 / ${GPUS_PER_JOB:-1}]
TOTAL_GPUS=$[$NUM_JOBS * ${GPUS_PER_JOB:-1}]
if [[ $TOTAL_GPUS -ge 8 ]]; then
    GREP_GPU=8
else
    GREP_GPU=$TOTAL_GPUS
fi

PORT=$(shuf -i 12000-20000 -n 1)
ROOT=..

if [[ ! -f $ROOT/configs/${CONF_NAME}.yaml ]]; then
    echo "config \`${CONF_NAME}\` not exist";
    exit 1
else
    CONF_FILE=${ROOT}/configs/${CONF_NAME}.yaml
    JOB_NAME=IDQ # ${CONF_NAME//\//_}
    echo "evaluating with configure \`$CONF_FILE\`"
fi

source r0.3.0

export PYTHONPATH=$ROOT:$PYTHONPATH

GLOG_vmodule=MemcachedClient=-1 srun \
  -p ${PART} \
  -n${NUM_JOBS} \
  --gres=gpu:${GREP_GPU} \
  --ntasks-per-node=${TASKS_PER_NODE} \
  --job-name=${JOB_NAME} \
  --mpi=pmi2 \
  --kill-on-bad-exit=1 \
  ${SLURM_EXTRA} \
python $ROOT/tools/_deprecated_train.py \
  --evaluate \
  --conf-path ${CONF_FILE} \
  --port ${PORT} \
  ${EXTRA}
