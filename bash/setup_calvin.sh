scp -r .cache/clip ~/.cache/
export PATH=$PATH:/mnt/bn/robotics/lxh/robot-flamingo
export PYTHONPATH=$PYTHONPATH:/mnt/bn/robotics/lxh/robot-flamingo

export http_proxy=http://bj-rd-proxy.byted.org:3128
export https_proxy=http://bj-rd-proxy.byted.org:3128

export CALVIN_ROOT=/mnt/bn/robotics/resources/calvin
export EVALUTION_ROOT=$(pwd)

# Install dependency for calvin
sudo apt-get -y install libegl1-mesa libegl1
sudo apt-get -y install libgl1

# Install dependency for dt
# sudo apt-get -y install libosmesa6-dev
# sudo apt-get -y install patchelf

# Install EGL mesa
sudo apt-get update -y -qq
sudo apt-get install -y -qq libegl1-mesa libegl1-mesa-dev
# pip install tacto numpy==1.19.5

#
# pip3 install open_flamingo

# install osmesa
# sudo wget https://github.com/mmatl/travis_debs/raw/master/xenial/mesa_18.3.3-0.deb
# sudo dpkg -i ./mesa_18.3.3-0.deb || true
# sudo apt install -f
# sudo git clone GitHub - mmatl/pyopengl: Repository for the PyOpenGL Project (LaunchPad Mirror)
# sudo pip install ./pyopengl
sudo apt install -y mesa-utils libosmesa6-dev llvm
sudo apt-get -y install meson
sudo apt-get -y build-dep mesa