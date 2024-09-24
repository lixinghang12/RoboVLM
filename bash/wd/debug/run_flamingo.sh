#!/usr/bin/env bash
. bash/wd/utils.sh

source /mnt/bn/robotics/resources/anaconda3_arnold/bin/activate calvin_mpt

CUR_DIR=$(cd $(dirname $0); pwd)
port=$(get_free_port)
free_gpu=$(get_free_gpu)
GPUS_PER_NODE=1
ARNOLD_WORKER_NUM=1
set -x
export http_proxy=http://bj-rd-proxy.byted.org:3128
export https_proxy=http://bj-rd-proxy.byted.org:3128
export CUDA_VISIBLE_DEVICES="$free_gpu"
export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=${ARNOLD_RDMA_DEVICE}
export NCCL_SOCKET_IFNAME=eth0
export BYTEPS_DDP_FIND_UNUSED_PARAMS=${BYTEPS_DDP_FIND_UNUSED_PARAMS:-0}

echo "GPU Device $free_gpu"
echo "Port $port"
echo "total workers: ${ARNOLD_WORKER_NUM}"
echo "cur worker id: ${ARNOLD_ID}"
echo "gpus per worker: ${ARNOLD_WORKER_GPU}"
echo "master ip: ${METIS_WORKER_0_HOST}"
echo "master port: ${port}"

subfix=`date "+%H-%M"`

echo "RUNNING:"
echo torchrun \
    --nnodes $ARNOLD_WORKER_NUM \
    --node_rank $ARNOLD_ID \
    --nproc_per_node $GPUS_PER_NODE \
    --master_addr $METIS_WORKER_0_HOST \
    --master_port $port \
    scripts/main.py \
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
    scripts/main.py \
    --exp_name ${subfix} \
    ${@:1} \
    --gpus $GPUS_PER_NODE \
    --num_nodes $ARNOLD_WORKER_NUM
