cd ..

git clone https://github.com/neheller/kits19.git
cd kits19

python -m starter_code.get_imaging

cd ..
git clone https://github.com/Ivan-Markic/kits19-challenge.git
mv kits19/data kits19-challenge/kits19

rm -rf kits19/

cd kits19-challenge

pip install -r requirements.txt

python conversion_data.py -d "kits19" -o "data"

pip install ipykernel
pip install wandb

wandb login

# Train first model which will do segmentation of kidney from whole ct image
python train_res_unet.py -e 20 -b 32 -l 0.0001 -g 2 -s 512 512 -d "data" --log "runs/ResUNet" --eval_intvl 5 --cp_intvl 5 --vis_intvl 5 --num_workers 2

# Using first model create ROI(region of interest) and save information in data/roi.json
# This information will be used by main model (dense unet model)
python get_roi.py -b 32 -g 2 -s 512 512 --org_data "kits19/data" --data "data" -r "runs/ResUNet/checkpoint/best.pth" -o "data/roi.json"

# Train model which will do segmentation of kidnety and tumor (create mask of image)
python train_dense_unet.py -e 20 -b 32 -l 0.0001 -g 2 -s 512 512 -d "data" --log "runs/DenseUNet" --eval_intvl 5 --cp_intvl 5 --vis_intvl 0 --num_workers 2