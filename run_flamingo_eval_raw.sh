#!/bin/bash
export http_proxy=http://bj-rd-proxy.byted.org:3128
export https_proxy=http://bj-rd-proxy.byted.org:3128

export CALVIN_ROOT=/mnt/bn/robotics/resources/calvin
export EVALUTION_ROOT=$(pwd)

# # Install dependency for calvin
# sudo apt-get -y install libegl1-mesa libegl1
# sudo apt-get -y install libgl1

# # # Install dependency for dt
# sudo apt-get -y install libosmesa6-dev
# sudo apt-get -y install patchelf

# # Copy clip weights
# mkdir -p ~/.cache/clip
# cp /mnt/bn/robotics/pretrained_models/clip/ViT-B-32.pt ~/.cache/clip

# Run
source /mnt/bn/robotics/resources/anaconda3_arnold/bin/activate calvin_mpt
pip3 install moviepy diffusers==0.29.1
pip3 install lightning==2.2.5
pip3 install transformers==4.36.2 -i "https://bytedpypi.byted.org/simple"

export MESA_GL_VERSION_OVERRIDE=4.1
cd $EVALUTION_ROOT
ckpt_dir=$1
config_path=$2
# sudo chmod 777 -R $ckpt_dir

GPUS_PER_NODE=4
# export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nnodes=1 --nproc_per_node=$GPUS_PER_NODE --master_port=6067 eval/calvin/evaluate_ddp-v2.py \
--config_path $config_path \
--ckpt_path $ckpt_dir \
--ckpt_idx 0 --raw_calvin


# bash run_flamingo_eval_raw.sh /mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/calvin_finetune/2024-06-24/default/epoch=1-step=47954.ckpt /mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/calvin_finetune/2024-06-24/default/2024-06-24_16:39:20.620112-project.json
# bash run_flamingo_eval_raw.sh /mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/flamingo_video/calvin_finetune/2024-07-18/17-25/epoch=0-step=10000.ckpt  /mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/flamingo_video/calvin_finetune/2024-07-18/17-25/2024-07-18_17:26:59.288620-project.json

# without history
# bash run_flamingo_eval_raw.sh /mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/calvin_finetune/2024-07-18/18-07/epoch=1-step=56566.ckpt /mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/calvin_finetune/2024-07-18/18-07/2024-07-18_18:08:35.738798-project.json
# bash run_flamingo_eval_raw.sh /mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/flamingo_video/calvin_finetune/2024-07-23/23-43/epoch=0-step=28283.ckpt /mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/flamingo_video/calvin_finetune/2024-07-23/23-43/2024-07-23_23:43:52.047089-project.json

# with history
# bash run_flamingo_eval_raw.sh /mnt/bn/robotics-data-lxh-lq/checkpoints/video_pretrain_manipulation/calvin_finetune/2024-07-18/23-49/epoch=1-step=159652.ckpt /mnt/bn/robotics-data-lxh-lq/logs/video_pretrain_manipulation/calvin_finetune/2024-07-18/23-49/2024-07-18_23:51:00.083905-project.json