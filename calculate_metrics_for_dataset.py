import click
import numpy as np
import wandb
import torch
import shutil
from pathlib2 import Path
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

import utils.checkpoint as cp
from dataset import KiTS19
from dataset.transform import MedicalTransform
from network import DenseUNet, SimpleUNet
from utils.metrics import Evaluator


@click.command()
@click.option('-b', '--batch', 'batch_size', help='Number of batch size', type=int, default=1, show_default=True)
@click.option('-g', '--num_gpu', help='Number of GPU', type=int, default=1, show_default=True)
@click.option('-s', '--size', 'img_size', help='Output image size', type=(int, int),
              default=(512, 512), show_default=True)
@click.option('-d', '--data', 'data_path', help='Path of kits19 data after conversion',
              type=click.Path(exists=True, dir_okay=True, resolve_path=True),
              default='data', show_default=True)
@click.option('--num_workers', help='Number of workers on dataloader. '
                                    'Recommend 0 in Windows. '
                                    'Recommend num_gpu in Linux',
              type=int, default=0, show_default=True)
@click.option('--type', help='Type of network',
              type=str, default='simple_unet', show_default=True)
def main(batch_size, num_gpu, img_size, data_path, num_workers, type):
    data_path = Path(data_path)

    # Make ROI usage conditional based on network type
    use_roi = type == 'dense_unet'
    roi_error_range = 15 if use_roi else 0
    transform = MedicalTransform(output_size=img_size, roi_error_range=roi_error_range, use_roi=use_roi)

    dataset = KiTS19(data_path, stack_num=3, spec_classes=[0, 1, 2], img_size=img_size,
                     use_roi=use_roi, roi_file='roi.json' if use_roi else None, 
                     roi_error_range=5 if use_roi else 0,
                     train_transform=transform, valid_transform=transform, test_transform=transform)

    # Initialize network based on type
    if type == 'dense_unet':
        net = DenseUNet(in_ch=dataset.img_channels, out_ch=dataset.num_classes)
        model_artifact = 'dense_unet_model_epoch_14:v0'
    elif type == 'simple_unet':
        net = SimpleUNet(in_ch=dataset.img_channels, out_ch=dataset.num_classes)
        model_artifact = 'simple_unet_model_epoch_15:v0'

    else:
        raise ValueError(f"Unknown network type: {type}")

    gpu_ids = [i for i in range(num_gpu)]

    # Initialize W&B
    run = wandb.init(
        project=type,
        name=f'metrics_calculation_{type}',
        config={
            "batch_size": batch_size,
            "architecture": net.__class__.__name__,
            "dataset": "kits19",
            "image_size": "512x512",
            "device:": f"cuda{str(gpu_ids)}"
        }
    )

    # Download the artifact
    artifact = run.use_artifact(model_artifact, type='model')
    artifact_dir = Path(artifact.download())

    # Define the save path
    resume = Path(f"runs/{net.__class__.__name__}/best/best.pth")
    resume.parent.mkdir(parents=True, exist_ok=True)

    print(f"Artifact downloaded to: {artifact_dir}")

    # Navigate to the renamed folder and find best.pth
    best_pth_file = artifact_dir / "best.pth"
    assert best_pth_file.exists(), f"File 'best.pth' not found inside {best_pth_file}"

    # Move best.pth to the save path
    shutil.move(best_pth_file, resume)
    print(f"Moved model file to: {resume}")

    if resume:
        data = {'net': net}
        cp_file = Path(resume)
        cp.load_params(data, cp_file, device='cpu')

    torch.cuda.empty_cache()

    net = torch.nn.DataParallel(net, device_ids=gpu_ids).cuda()

    net.eval()
    torch.set_grad_enabled(False)
    transform.eval()

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
