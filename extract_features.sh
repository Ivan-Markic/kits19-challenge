# /bin/bash

python conversion_data.py
python eval_unet.py -b 12 -g 2 
python extract_pyradiomics_features_to_csv.py