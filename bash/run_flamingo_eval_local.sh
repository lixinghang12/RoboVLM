export PATH=$PATH:/mnt/bn/robotics-data-lxh-lq/RoboVLM
export PYTHONPATH=$PYTHONPATH:/mnt/bn/robotics-data-lxh-lq/RoboVLM

export http_proxy=http://bj-rd-proxy.byted.org:3128
export https_proxy=http://bj-rd-proxy.byted.org:3128

export CALVIN_ROOT=/mnt/bn/robotics/resources/calvin
export EVALUTION_ROOT=$(pwd)

# Install dependency for calvin
sudo apt-get -y install libegl1-mesa libegl1
sudo apt-get -y install libgl1

# Install EGL mesa
sudo apt-get update -y -qq
sudo apt-get install -y -qq libegl1-mesa libegl1-mesa-dev 
# sudo apt-get install -y libglfw3-dev libgles2-mesa-dev

# source /mnt/bn/robotics/resources/anaconda3_arnold/bin/activate calvin_mpt

sudo apt install -y mesa-utils libosmesa6-dev llvm
sudo apt-get install -y meson
sudo apt-get build-dep mesa

# Copy clip weights
mkdir -p ~/.cache/clip
cp /mnt/bn/robotics-data-lxh-lq/RoboVLM/.cache/clip/ViT-L-14.pt ~/.cache/clip

# Run
pip3 install moviepy

export MESA_GL_VERSION_OVERRIDE=4.1

source /mnt/bn/robotics/resources/anaconda3_arnold/bin/activate calvin_mpt
cd $EVALUTION_ROOT

START_CKPT_IDX=$1
NUM_MODELS=$2
echo START_CKPT_IDX=$START_CKPT_IDX
echo NUM_MODELS=$NUM_MODELS

# sudo chmod 777 -R $CKPT_DIR

END_CKPT_IDX=`expr ${NUM_MODELS} + $START_CKPT_IDX`
for ((i=${START_CKPT_IDX};i<${END_CKPT_IDX};i++)); do
    echo evaluating checkpoint $i
    echo python3 eval/calvin/evaluate.py \
        --ckpt_idx $i \
        ${@:3}
    python3 eval/calvin/evaluate.py \
        --ckpt_idx $i \
        ${@:3}
done
