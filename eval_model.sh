#!/bin/bash

# Todo: set best model from 20 epoch traning on this path.
python calculate_metrics_for_dataset.py -b 32 -g 2 -s 512 512 -d "data" -r "runs/DenseUNet/checkpoint/best.pth" --num_workers 2

# This should generate predict masks for each case inside kits19 dataset
python eval_dense_unet.py -b 32 -g 2 -s 512 512 -d "data" -r "runs/DenseUNet/checkpoint/best.pth" --num_workers 2 -o "kits19"

# Install this so python file extract_pyradiomics_features_to_csv.py can be runned
pip install SimpleITK

pip install pyradiomics

# Run extract_pyradiomics_features_to_csv which should create radiomics_features_from_predicted_masks.csv
python extract_pyradiomics_features_to_csv