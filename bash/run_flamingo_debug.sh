#!/usr/bin/env bash

export http_proxy=http://bj-rd-proxy.byted.org:3128
export https_proxy=http://bj-rd-proxy.byted.org:3128

source /mnt/bn/robotics/resources/anaconda3_arnold/bin/activate calvin_mpt
bash setup.sh

CUR_DIR=$(cd $(dirname $0); pwd)
# cd $CUR_DIR/../

#LEGACY DDP CONFIGS
ports=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
port=${ports[0]}

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

# convert deepspeed checkpoint first
if [ $ARNOLD_ID == "0" ]; then
  echo "---------- Converting deepspeed checkpoint to fp32. ----------"
  python3 tools/convert_deepspeed_to_fp32.py ${@:1}
fi

echo "RUNNING:"
echo torchrun \
    --nnodes $ARNOLD_WORKER_NUM \
    --node_rank $ARNOLD_ID \
    --nproc_per_node 1 \
    --master_addr $METIS_WORKER_0_HOST \
    --master_port $port \
    scripts/main_flamingo.py \
    ${@:1} \
    --gpus 1 \
    --num_nodes $ARNOLD_WORKER_NUM

torchrun \
    --nnodes $ARNOLD_WORKER_NUM \
    --node_rank $ARNOLD_ID \
    --nproc_per_node 1 \
    --master_addr $METIS_WORKER_0_HOST \
    --master_port $port \
    scripts/main_flamingo.py \
    ${@:1} \
    --gpus 1 \
    --num_nodes $ARNOLD_WORKER_NUM

