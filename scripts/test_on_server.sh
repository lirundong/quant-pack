#!/usr/bin/env bash
set -e

while getopts :p:qc:x:m:n: opt
do
    case "$opt" in
    p) PART=$OPTARG ;;
    q) EXTRA="${EXTRA} --quant";;
    c) CONF_NAME=$OPTARG;;
    x) EXTRA="${EXTRA} --extra ${OPTARG}";;
    m) EXTRA="${EXTRA} --comment ${OPTARG}";;
    n) NUM_JOBS=$OPTARG
       if [[ $OPTARG -ge 8 ]]; then
            GREP_GPU=8
       else
            GREP_GPU=$OPTARG
       fi;;
    *) echo "invalid flag -$opt";
       exit 1;;
    esac
done

T=$(date +%Y%m%d-%H-%M-%S)
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
  --ntasks-per-node=${GREP_GPU} \
  --job-name=${JOB_NAME} \
  --mpi=pmi2 \
  --kill-on-bad-exit=1 \
  ${SLURM_EXTRA} \
python $ROOT/tools/train_val_classifier.py \
  --evaluate \
  --conf-path ${CONF_FILE} \
  --port ${PORT} \
  ${EXTRA}
