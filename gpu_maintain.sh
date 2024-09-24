#!/bin/bash
num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
root_dir="/mnt/bn/robotics-data-lxh-lq/RoboVLM"

# environment_script="scripts/environment.sh"

for i in $(seq 0 $((num_gpus - 1))); do  
  session_name="maintain_gpu_$i"
  tmux new-session -d -s $session_name
  tmux send-keys -t $session_name "cd $root_dir" C-m
  tmux send-keys -t $session_name "export CUDA_VISIBLE_DEVICES=$i" C-m
  tmux send-keys -t $session_name "source /mnt/bn/robotics/resources/anaconda3_arnold/bin/activate robovlm_openvla" C-m
  tmux send-keys -t $session_name "python gpu_running.py" C-m
done