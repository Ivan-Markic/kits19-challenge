#!/bin/bash

# Train model which will do segmentation of kidney and tumor (create mask of image)
python train_unet.py -e 20 -b 20 -l 0.0001 -g 2 -s 512 512 -d "data" --log "runs/SimpleUNet" --eval_intvl 1 --cp_intvl 5 --vis_intvl 0 --num_workers 2 --type simple_unet
