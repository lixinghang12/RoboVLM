export http_proxy=http://bj-rd-proxy.byted.org:3128
export https_proxy=http://bj-rd-proxy.byted.org:3128

sudo chmod 777 -R /mnt/bn/robotics-data-lxh-lq/logs
sudo chmod 777 -R /mnt/bn/robotics-data-lxh-lq/checkpoints
sudo chmod 777 -R /mnt/bn/robotics-data-lxh-lq/cache

source /mnt/bn/robotics/resources/anaconda3_arnold/bin/activate calvin_mpt

sudo apt-get update
sudo apt install -y screen
sudo apt-get -y install libosmesa6-dev
sudo apt-get -y install patchelf

pip3 install lightning==2.1.0
pip3 install deepspeed==0.12.3
pip3 install transformations
pip3 install thop

pip3 install transformers==4.37.2 tokenizers==0.15.1
pip3 install diffusers==0.29.1
