import click
import numpy as np
import wandb
import torch
from pathlib2 import Path
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm

import utils.checkpoint as cp
from dataset import KiTS19
from dataset.transform import MedicalTransform
from network import DenseUNet
from utils.metrics import Evaluator


@click.command()
@click.option('-b', '--batch', 'batch_size', help='Number of batch size', type=int, default=1, show_default=True)
@click.option('-g', '--num_gpu', help='Number of GPU', type=int, default=1, show_default=True)
@click.option('-s', '--size', 'img_size', help='Output image size', type=(int, int),
              default=(512, 512), show_default=True)
@click.option('-d', '--data', 'data_path', help='Path of kits19 data after conversion',
              type=click.Path(exists=True, dir_okay=True, resolve_path=True),
              default='data', show_default=True)
@click.option('-r', '--resume', help='Resume model',
              type=click.Path(exists=True, file_okay=True, resolve_path=True), required=True)
@click.option('--num_workers', help='Number of workers on dataloader. '
                                    'Recommend 0 in Windows. '
                                    'Recommend num_gpu in Linux',
              type=int, default=0, show_default=True)
def main(batch_size, num_gpu, img_size, data_path,
         resume, num_workers):
    data_path = Path(data_path)

    transform = MedicalTransform(output_size=img_size, roi_error_range=15, use_roi=True)

    dataset = KiTS19(data_path, stack_num=3, spec_classes=[0, 1, 2], img_size=img_size,
                     use_roi=True, roi_file='roi.json', roi_error_range=5,
                     train_transform=transform, valid_transform=transform, test_transform=transform)

    net = DenseUNet(in_ch=dataset.img_channels, out_ch=dataset.num_classes)

    if resume:
        data = {'net': net}
        cp_file = Path(resume)
        cp.load_params(data, cp_file, device='cpu')

    gpu_ids = [i for i in range(num_gpu)]

    wandb.init(
        # set the wandb project where this run will be logged
        project="best_dense_unet_eval",
        name=f'20_epoch_dense_unet',

        # track hyperparameters and run metadata
        config={
            "batch_size": batch_size,
            "architecture": net.__class__.__name__,
            "dataset": "kits19",
            "image_size": "512x512",
            "device:": f"cuda{str(gpu_ids)}"
        }
    )

    torch.cuda.empty_cache()

    net = torch.nn.DataParallel(net, device_ids=gpu_ids).cuda()

    evaluation(net, dataset, batch_size, num_workers, wandb, 'train')
    evaluation(net, dataset, batch_size, num_workers, wandb, 'valid')

    wandb.finish()

def evaluation(net, dataset, batch_size, num_workers, wandb, type):
    type = type.lower()
    if type == 'train':
        subset = dataset.train_dataset
        case_slice_indices = dataset.train_case_slice_indices
    elif type == 'valid':
        subset = dataset.valid_dataset
        case_slice_indices = dataset.valid_case_slice_indices

    sampler = SequentialSampler(subset)
    data_loader = DataLoader(subset, batch_size=batch_size, sampler=sampler,
                             num_workers=num_workers, pin_memory=True)
    evaluator = Evaluator(dataset.num_classes)

    case = 0
    vol_label = []
    vol_output = []

    with tqdm(total=len(case_slice_indices) - 1, ascii=True, desc=f'eval/{type:5}', dynamic_ncols=True) as pbar:
        for batch_idx, data in enumerate(data_loader):
            imgs, labels, idx = data['image'].cuda(), data['label'], data['index']

            outputs = net(imgs)
            predicts = outputs['output']
            predicts = predicts.argmax(dim=1)

            labels = labels.cpu().detach().numpy()
            predicts = predicts.cpu().detach().numpy()
            idx = idx.numpy()

            vol_label.append(labels)
            vol_output.append(predicts)

            while case < len(case_slice_indices) - 1 and idx[-1] >= case_slice_indices[case + 1] - 1:
                vol_output = np.concatenate(vol_output, axis=0)
                vol_label = np.concatenate(vol_label, axis=0)

                vol_num_slice = case_slice_indices[case + 1] - case_slice_indices[case]
                evaluator.add(vol_output[:vol_num_slice], vol_label[:vol_num_slice])

                vol_output = [vol_output[vol_num_slice:]]
                vol_label = [vol_label[vol_num_slice:]]
                case += 1
                pbar.update(1)

    metrics = evaluator.eval()

    num_classes = dataset.num_classes

    # Log per-case metrics as tables
    wandb.log({
        f"{type}/Dice Per Case": wandb.Table(data=metrics['dc_each_case'],
                                             columns=[f"Class {i}" for i in range(num_classes)]),
        f"{type}/Accuracy Per Case": wandb.Table(data=metrics['acc_each_case'],
                                                 columns=[f"Class {i}" for i in range(num_classes)]),
        f"{type}/IoU Per Case": wandb.Table(data=metrics['iou_each_case'],
                                            columns=[f"Class {i}" for i in range(num_classes)]),
    })


if __name__ == '__main__':
    main()
