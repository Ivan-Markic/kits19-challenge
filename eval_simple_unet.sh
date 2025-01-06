#!/bin/bash

python calculate_metrics_for_dataset.py -b 12 -g 2 -s 512 512 -d "data" --num_workers 2 --type simple_unet

