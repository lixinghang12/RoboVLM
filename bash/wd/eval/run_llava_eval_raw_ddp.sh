#!/bin/bash
export http_proxy=http://bj-rd-proxy.byted.org:3128
export https_proxy=http://bj-rd-proxy.byted.org:3128

export CALVIN_ROOT=/mnt/bn/robotics/resources/calvin
export EVALUTION_ROOT=$(pwd)

# # Install dependency for calvin
sudo apt-get -y install libegl1-mesa libegl1
sudo apt-get -y install libgl1

# # Install dependency for dt
sudo apt-get -y install libosmesa6-dev
sudo apt-get -y install patchelf

# # Copy clip weights
# mkdir -p ~/.cache/clip
# cp /mnt/bn/robotics/pretrained_models/clip/ViT-B-32.pt ~/.cache/clip

# Run
source /mnt/bn/robotics/resources/anaconda3_arnold/bin/activate robollava
pip3 install moviepy diffusers==0.29.1
pip3 install lightning==2.2.5
pip3 install transformers==4.36.2 -i "https://bytedpypi.byted.org/simple"

export MESA_GL_VERSION_OVERRIDE=4.1
cd $EVALUTION_ROOT
ckpt_dir=$1
config_path=$2
sudo chmod 777 -R $ckpt_dir
# TODO delete this for debug
# export CUDA_VISIBLE_DEVICES=7

python3 eval/calvin/evaluate_ddp.py --config_path $config_path \
--ckpt_path $ckpt_dir \
--ckpt_idx 0 --raw_calvin
