#!/bin/bash
export http_proxy=http://bj-rd-proxy.byted.org:3128
export https_proxy=http://bj-rd-proxy.byted.org:3128

export CALVIN_ROOT=/mnt/bn/robotics/resources/calvin
export EVALUTION_ROOT=$(pwd)

# Install dependency for calvin
sudo apt-get -y install libegl1-mesa libegl1
sudo apt-get -y install libgl1

# Install dependency for dt
sudo apt-get -y install libosmesa6-dev
sudo apt-get -y install patchelf

# Copy clip weights
mkdir -p ~/.cache/clip
cp /mnt/bn/robotics/pretrained_models/clip/ViT-B-32.pt ~/.cache/clip

# Run
source /mnt/bn/robotics/resources/anaconda3_arnold/bin/activate calvin_mpt
pip3 install moviepy
pip3 install transformers==4.38.1 -i "https://bytedpypi.byted.org/simple"

export MESA_GL_VERSION_OVERRIDE=3.3
cd $EVALUTION_ROOT

START_CKPT_IDX=$1
NUM_MODELS=$2
echo START_CKPT_IDX=$START_CKPT_IDX
echo NUM_MODELS=$NUM_MODELS

sudo chmod 777 -R $CKPT_DIR

END_CKPT_IDX=`expr ${NUM_MODELS} + $START_CKPT_IDX`
for ((i=${START_CKPT_IDX};i<${END_CKPT_IDX};i++)); do
    echo evaluating checkpoint $i
    echo python3 evaluation/calvin/evaluate.py \
        --ckpt_idx $i \
        ${@:3}
    python3 evaluation/calvin/evaluate.py \
        --ckpt_idx $i \
        ${@:3}
done
