#!/bin/bash

python calculate_metrics_for_dataset.py -b 8 -g 2 -s 512 512 -d "data" --num_workers 2

# This should generate predict masks for each case inside kits19 dataset
python eval_unet.py -b 8 -g 2 -s 512 512 -d "data" -r "runs/SimpleUNet/best.pth" --num_workers 2 -o "kits19" --type simple_unet

# Install this so python file extract_pyradiomics_features_to_csv.py can be runned
pip install SimpleITK

pip install pyradiomics

# Run extract_pyradiomics_features_to_csv which should create radiomics_features_from_predicted_masks.csv
python extract_pyradiomics_features_to_csv.py