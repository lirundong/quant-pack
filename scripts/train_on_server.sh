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
    esac
done

T=`date +%Y%m%d-%H-%M-%S`
ROOT=..

if [[ ! -f $ROOT/configs/${CONF_NAME}.yaml ]]; then
    echo "config \`${CONF_NAME}\` not exist";
    JOB_NAME=${CONF_NAME/\//_}
    exit -1
else
    CONF_FILE=${ROOT}/configs/${CONF_NAME}.yaml
    echo "training with configure \`$CONF_FILE\`"
fi

source $HOME/.local/bin/env_spring.sh
source activate r0.2.2

export PYTHONPATH=$ROOT:$PYTHONPATH

GLOG_vmodule=MemcachedClient=-1 srun \
   --mpi=pmi2 -p ${PART} -n${NUM_JOBS} \
   --gres=gpu:${GREP_GPU} --ntasks-per-node=${GREP_GPU} \
   --job-name=${JOB_NAME} \
   --kill-on-bad-exit=1 \
python $ROOT/tools/train_val_classifier.py \
  --conf-path ${CONF_FILE} \
  ${EXTRA} \
  2>&1 | tee train_${JOB_NAME}_${T}.log
