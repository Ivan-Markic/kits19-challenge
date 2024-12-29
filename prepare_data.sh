#!/bin/bash

# Installing torch version for cuda usage
conda install pytorch==1.10.0 torchvision==0.11.0 torchaudio==0.10.0 cudatoolkit=10.2 -c pytorch

git clone https://github.com/Ivan-Markic/kits19-challenge.git
pip install -r kits19-challenge/requirements.txt

wandb login $WANDB_API_KEY

git clone https://github.com/neheller/kits19.git
cd kits19

python -m starter_code.get_imaging

cd ..
mv kits19/data kits19-challenge/kits19

rm -rf kits19/

cd kits19-challenge

python conversion_data.py -d "kits19" -o "data"