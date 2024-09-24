#!/usr/bin/env bash

export http_proxy=http://bj-rd-proxy.byted.org:3128
export https_proxy=http://bj-rd-proxy.byted.org:3128

source /mnt/bn/robotics/resources/anaconda3_arnold/bin/activate robollava
pip install lightning==2.2.5
pip install transformers==4.36.2 tokenizers==0.15.0
pip install torch==2.3.1
pip install diffusers

scp -r /mnt/bn/robotics-data-lxh-lq/RoboVLM/.cache/clip ~/.cache
# bash setup.sh

CUR_DIR=$(cd $(dirname $0); pwd)
# sudo chmod 777 -R /mnt/bn/robotics-data-lxh-lq/LLaVA
# cd /mnt/bn/robotics-data-lxh-lq/LLaVA
# pip install -e .
# pip install -e ".[train]"
# pip install flash-attn --no-build-isolation
# cd $CUR_DIR

#LEGACY DDP CONFIGS
ports=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
port=${ports[0]}

echo "total workers: ${ARNOLD_WORKER_NUM}"
echo "cur worker id: ${ARNOLD_ID}"
echo "gpus per worker: ${ARNOLD_WORKER_GPU}"
echo "master ip: ${METIS_WORKER_0_HOST}"
echo "master port: ${port}"

set -x
export PYTHONUNBUFFERED=1
export BYTED_TORCH_FX=O1
export BYTED_TORCH_BYTECCL=${BYTED_TORCH_BYTECCL:-O1}

# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_LEVEL=LOC
# export NCCL_SOCKET_IFNAME=<your_network_interface>

export OMP_NUM_THREADS=16
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=${ARNOLD_RDMA_DEVICE}
export NCCL_SOCKET_IFNAME=eth0
export BYTEPS_DDP_FIND_UNUSED_PARAMS=${BYTEPS_DDP_FIND_UNUSED_PARAMS:-0}

# export CUDA_LAUNCH_BLOCKING=
# ARNOLD_WORKER_NUM=1
# ARNOLD_ID=0
GPUS_PER_NODE=$ARNOLD_WORKER_GPU
# GPUS_PER_NODE=2
# convert deepspeed checkpoint first
if [ $ARNOLD_ID == "0" ]; then
  echo "---------- Converting deepspeed checkpoint to fp32. ----------"
  python3 tools/convert_deepspeed_to_fp32.py ${@:1}
fi

subfix=`date "+%H-%M"`

echo "RUNNING:"
echo torchrun \
    --nnodes $ARNOLD_WORKER_NUM \
    --node_rank $ARNOLD_ID \
    --nproc_per_node $GPUS_PER_NODE \
    --master_addr $METIS_WORKER_0_HOST \
    --master_port $port \
    scripts/main_llava.py \
    --exp_name ${subfix} \
    ${@:1} \
    --gpus $GPUS_PER_NODE \
    --num_nodes $ARNOLD_WORKER_NUM

torchrun \
    --nnodes $ARNOLD_WORKER_NUM \
    --node_rank $ARNOLD_ID \
    --nproc_per_node $GPUS_PER_NODE \
    --master_addr $METIS_WORKER_0_HOST \
    --master_port $port \
    scripts/main_llava.py \
    --exp_name ${subfix} \
    ${@:1} \
    --gpus $GPUS_PER_NODE \
    --num_nodes $ARNOLD_WORKER_NUM

