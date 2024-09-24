#!/usr/bin/env bash

export http_proxy=http://bj-rd-proxy.byted.org:3128
export https_proxy=http://bj-rd-proxy.byted.org:3128

source /mnt/bn/robotics/resources/anaconda3_arnold/bin/activate robollava
# source /mnt/bn/robotics-data-lxh-lq/anaconda3/bin/activate robollava
# pip install lightning==2.2.5
# pip install transformers==4.37.2 tokenizers==0.15.1
# bash setup.sh

CUR_DIR=$(cd $(dirname $0); pwd)

# pip install -e .
# pip install -e ".[train]"
# pip install flash-attn --no-build-isolation
# cd $CUR_DIR

#LEGACY DDP CONFIGS

while :
do
    port="`shuf -i 2000-65000 -n 1`"
    netstat -tuln | grep -q ":$PORT "
    if [[ $? -eq 1 ]]; then
        echo "Free port: $port"
        break
    fi
done
echo $port

# ports=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
# port=${ports[0]}

echo "total workers: ${ARNOLD_WORKER_NUM}"
echo "cur worker id: ${ARNOLD_ID}"
echo "gpus per worker: ${ARNOLD_WORKER_GPU}"
echo "master ip: ${METIS_WORKER_0_HOST}"
echo "master port: ${port}"

export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=${ARNOLD_RDMA_DEVICE}
export NCCL_SOCKET_IFNAME=eth0
export BYTEPS_DDP_FIND_UNUSED_PARAMS=${BYTEPS_DDP_FIND_UNUSED_PARAMS:-0}
# export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"

# GPUS_PER_NODE=$ARNOLD_WORKER_GPU
GPUS_PER_NODE=1

# convert deepspeed checkpoint first
if [ $ARNOLD_ID == "0" ]; then
  echo "---------- Converting deepspeed checkpoint to fp32. ----------"
  python3 tools/convert_deepspeed_to_fp32.py ${@:1}
fi

echo "RUNNING:"
echo torchrun \
    --nnodes $ARNOLD_WORKER_NUM \
    --node_rank $ARNOLD_ID \
    --nproc_per_node $GPUS_PER_NODE \
    --master_addr $METIS_WORKER_0_HOST \
    --master_port $port \
    scripts/main_qwen.py \
    ${@:1} \
    --gpus $GPUS_PER_NODE \
    --num_nodes $ARNOLD_WORKER_NUM

torchrun \
    --nnodes $ARNOLD_WORKER_NUM \
    --node_rank $ARNOLD_ID \
    --nproc_per_node $GPUS_PER_NODE \
    --master_addr $METIS_WORKER_0_HOST \
    --master_port $port \
    scripts/main_qwen.py \
    ${@:1} \
    --gpus $GPUS_PER_NODE \
    --num_nodes $ARNOLD_WORKER_NUM

