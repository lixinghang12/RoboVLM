#!/usr/bin/env bash

# source /mnt/bn/robotics/resources/anaconda3_arnold/bin/activate calvin_mpt

sudo apt-get update
sudo apt install -y screen
sudo apt-get -y install libosmesa6-dev
sudo apt-get -y install patchelf

#pip3 install torch==2.1.0 torchvision torchaudio -i "https://bytedpypi.byted.org/simple"
pip3 install lightning==2.1.0
pip3 install deepspeed==0.12.3
pip3 install transformations
pip3 install torchvision torchaudio thop

pip3 install transformers==4.33.3 -i "https://bytedpypi.byted.org/simple"
pip3 install packaging==21.3 -i "https://bytedpypi.byted.org/simple"
pip3 install tensorboard -i "https://bytedpypi.byted.org/simple"

pip3 install ftfy regex tqdm -i "https://bytedpypi.byted.org/simple"
pip3 install matplotlib decord pandas -i "https://bytedpypi.byted.org/simple"
pip3 install ipdb -i "https://bytedpypi.byted.org/simple"

mkdir -p ~/.cache
cp -r /mnt/bn/robotics-data-hl/lxh/RoboFlamingoV2/.cache/clip ~/.cache/
