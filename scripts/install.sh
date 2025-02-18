#!/usr/bin/env bash
conda create -n whac python=3.8 -y
conda activate whac

# STEP 1: Install dependencies for SMPLest-X and DPVO
conda install pytorch==1.12.0 torchvision==0.13.0 torchaudio==0.12.0 cudatoolkit=11.3 -c pytorch -y
pip install -r requirements.txt

# osmesa
apt-get update
apt-get install python-opengl -y
apt-get install libosmesa6 -y 

# pytorch3d
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py38_cu113_pyt1120/download.html

# update version for pyopengl
# Note: please ignore the incompatible error message if 3.1.4 can be installed
pip install pyopengl==3.1.4

# STEP 2: Setup DPVO
cd ./third_party/DPVO
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
unzip eigen-3.4.0.zip -d thirdparty && rm -rf eigen-3.4.0.zip
wget https://www.dropbox.com/s/nap0u8zslspdwm4/models.zip 
unzip models.zip -d pretrained_models && rm -rf models.zip
conda install pytorch-scatter=2.0.9 -c rusty1s
pip install .

# STEP 3: Setup SMPLest-X
# prepare according to the README.md in the SMPLest-X repo