#!/bin/bash

source /data/home/hanbo/anaconda3/bin/activate flamingo
echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH
GPUS_PER_NODE=$1
echo GPUS_PER_NODE=$GPUS_PER_NODE

# echo "---------- Converting deepspeed checkpoint to fp32. ----------"
# srun -u --mem=40000 --gres=gpu:1 --cpus-per-task=8 --job-name "convert_ckpt" python3 tools/convert_deepspeed_to_fp32.py ${@:1}
COMMAND="torchrun \
    --nnodes 1 \
    --nproc_per_node $GPUS_PER_NODE \
    scripts/main_flamingo.py \
    ${@:2} \
    --gpus $GPUS_PER_NODE \
    --num_nodes 1"

echo $COMMAND
eval $COMMAND

