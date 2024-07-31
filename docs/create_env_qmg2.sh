#!/bin/bash

conda create --name qmg2 python=3.9 -y
conda activate qmg2
conda install ipykernel -y
python -m ipykernel install --user --name qmg2 --display-name "qmg2"
pip install pennylane pennylane-lightning-gpu
pip install torch==2.2.0
pip install cuquantum
python -m pip install nvidia-cusparse-cu11 nvidia-cublas-cu11 nvidia-cuda-runtime-cu11 custatevec-cu11

pip install rdkit
pip install pandas
pip install matplotlib # ==3.9.1
pip install pylatexenc
# pip install nvidia-cusparse-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 custatevec_cu12
# pip install custatevec_cu12
pip install scikit-learn
# pip install torchvision==0.17.0

pip install tqdm
