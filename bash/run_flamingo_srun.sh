#!/bin/bash

source /data/home/hanbo/anaconda3/bin/activate robovlm
GPUS_PER_NODE=$1
CPU_PER_NODE=$(($GPUS_PER_NODE * 4))
MEMORY=$(($GPUS_PER_NODE * 30 * 1024))
echo GPUS_PER_NODE=$GPUS_PER_NODE
echo CPU_PER_NODE=$CPU_PER_NODE
echo MEMORY=$MEMORY

# echo "---------- Converting deepspeed checkpoint to fp32. ----------"
# srun -u --mem=40000 --gres=gpu:1 --cpus-per-task=8 --job-name "convert_ckpt" python3 tools/convert_deepspeed_to_fp32.py ${@:1}
COMMAND="srun \
  --job-name=hanbo-vlm \
  --nodes=1 \
  --ntasks-per-node=1 \
  --cpus-per-task="$CPU_PER_NODE" \
  --mem="$MEMORY"M \
  --gres=gpu:"$GPUS_PER_NODE" \
  --output=/data/home/hanbo/projects/slurm_logs/%x-%j.out \
  torchrun \
  --nproc_per_node "$GPUS_PER_NODE" \
  test_slurm.py"

echo $COMMAND
eval $COMMAND

