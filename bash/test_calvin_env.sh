#!/bin/bash
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
# pip3 install moviepy

export MESA_GL_VERSION_OVERRIDE=4.1
# cd $EVALUTION_ROOT

source /mnt/bn/robotics/resources/anaconda3_arnold/bin/activate calvin_mpt
# /mnt/bn/robotics/resources/anaconda3_arnold/condabin/conda install -y osmesa
# sudo pip3 uninstall pyopengl pyrender && pip3 install pyopengl pyrender

# sudo pip uninstall pyopengl pyrender && pip install pyopengl pyrender

python eval/calvin/env_test.py