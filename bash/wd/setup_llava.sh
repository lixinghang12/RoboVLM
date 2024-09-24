export http_proxy=http://bj-rd-proxy.byted.org:3128
export https_proxy=http://bj-rd-proxy.byted.org:3128

sudo chmod 777 -R /mnt/bn/robotics-data-lxh-lq/logs
sudo chmod 777 -R /mnt/bn/robotics-data-lxh-lq/checkpoints
sudo chmod 777 -R /mnt/bn/robotics-data-lxh-lq/cache

source /mnt/bn/robotics/resources/anaconda3_arnold/bin/activate robollava
pip install lightning==2.1.0
pip install transformers==4.36.2 tokenizers==0.15.0
pip install torch==2.3.1
pip install diffusers
pip install flash_attn pytorchvideo opencv-python
sudo apt -y install libgl1-mesa-glx
