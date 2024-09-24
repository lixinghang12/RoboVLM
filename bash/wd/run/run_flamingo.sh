#!/usr/bin/env bash
export http_proxy=http://bj-rd-proxy.byted.org:3128
export https_proxy=http://bj-rd-proxy.byted.org:3128

sudo chmod 777 -R /mnt/bn/robotics-data-lxh-lq/logs
sudo chmod 777 -R /mnt/bn/robotics-data-lxh-lq/checkpoints
sudo chmod 777 -R /mnt/bn/robotics-data-lxh-lq/cache

source /mnt/bn/robotics/resources/anaconda3_arnold/bin/activate calvin_mpt
CUR_DIR=$(cd $(dirname $0); pwd)

ports=(`echo $METIS_WORKER_0_PORT | tr ',' ' '`)
port=${ports[0]}
GPUS_PER_NODE=$ARNOLD_WORKER_GPU

echo "GPU Device $free_gpu"
echo "Port $port"
echo "total workers: ${ARNOLD_WORKER_NUM}"
echo "cur worker id: ${ARNOLD_ID}"
echo "gpus per worker: ${ARNOLD_WORKER_GPU}"
echo "master ip: ${METIS_WORKER_0_HOST}"
echo "master port: ${port}"

set -x
export PYTHONUNBUFFERED=1
export NCCL_IB_GID_INDEX=3
export BYTED_TORCH_FX=O1
export BYTED_TORCH_BYTECCL=${BYTED_TORCH_BYTECCL:-O1}

export OMP_NUM_THREADS=16
export NCCL_IB_DISABLE=0
export NCCL_IB_HCA=${ARNOLD_RDMA_DEVICE}
export NCCL_SOCKET_IFNAME=eth0
export BYTEPS_DDP_FIND_UNUSED_PARAMS=${BYTEPS_DDP_FIND_UNUSED_PARAMS:-0}

convert deepspeed checkpoint first
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
