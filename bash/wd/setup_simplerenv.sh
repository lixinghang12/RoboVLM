sudo apt-get install -yqq --no-install-recommends libvulkan-dev vulkan-tools
sudo mkdir -p /usr/share/vulkan/icd.d
sudo wget -q -P /usr/share/vulkan/icd.d https://raw.githubusercontent.com/haosulab/ManiSkill/main/docker/nvidia_icd.json
sudo wget -q -O /usr/share/glvnd/egl_vendor.d/10_nvidia.json https://raw.githubusercontent.com/haosulab/ManiSkill/main/docker/10_nvidia.json

export http_proxy=http://bj-rd-proxy.byted.org:3128
export https_proxy=http://bj-rd-proxy.byted.org:3128

source /mnt/bn/robotics/resources/anaconda3_arnold/bin/activate robovlm_openvla
pip install lightning==2.2.5
pip install transformers==4.37.2 tokenizers==0.15.1
pip install diffusers
pip install decord