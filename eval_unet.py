import click
import cv2
import nibabel as nib
import numpy as np
import torch
from pathlib2 import Path
from torch.utils.data import DataLoader, SequentialSampler
from tqdm import tqdm

import utils.checkpoint as cp
from dataset import KiTS19
from dataset.transform import MedicalTransform
from network import DenseUNet, SimpleUNet

@click.command()
@click.option('-b', '--batch', 'batch_size', help='Number of batch size', type=int, default=12, show_default=True)
@click.option('-g', '--num_gpu', help='Number of GPU', type=int, default=2, show_default=True)
@click.option('-s', '--size', 'img_size', help='Output image size', type=(int, int),
              default=(512, 512), show_default=True)
@click.option('-d', '--data', 'data_path', help='Path of kits19 data after conversion',
              type=click.Path(exists=True, dir_okay=True, resolve_path=True),
              default='data', show_default=True)
@click.option('-r', '--resume', help='Resume model',
              type=click.Path(exists=True, file_okay=True, resolve_path=True), default='runs/SimpleUNet/best/best.pth', show_default=True)
@click.option('-o', '--output', 'output_path', help='output image path',
              type=click.Path(dir_okay=True, resolve_path=True), default='kits19', show_default=True)
@click.option('--num_workers', help='Number of workers on dataloader. '
                                    'Recommend 0 in Windows. '
                                    'Recommend num_gpu in Linux',
              type=int, default=2, show_default=True)
@click.option('--type', help='Type of network',
              type=str, default='simple_unet', show_default=True)
def main(batch_size, num_gpu, img_size, data_path, resume, output_path, num_workers, type):
    data_path = Path(data_path)
    output_path = Path(output_path)
    if not output_path.exists():
        output_path.mkdir(parents=True)
    
    # Make ROI usage conditional based on network type
    use_roi = False
    roi_error_range = 0
    transform = MedicalTransform(output_size=img_size, roi_error_range=roi_error_range, use_roi=use_roi)
    
    dataset = KiTS19(data_path, stack_num=3, spec_classes=[0, 1, 2], img_size=img_size,
                     use_roi=use_roi, roi_file=None, 
                     roi_error_range=roi_error_range,
                     train_transform=transform, valid_transform=transform, test_transform=transform)
    
    if type == 'dense_unet':
        net = DenseUNet(in_ch=dataset.img_channels, out_ch=dataset.num_classes)
    elif type == 'simple_unet':
        net = SimpleUNet(in_ch=dataset.img_channels, out_ch=dataset.num_classes)
    
    if resume:
        data = {'net': net}
        cp_file = Path(resume)
        cp.load_params(data, cp_file, device='cpu')
    
    gpu_ids = [i for i in range(num_gpu)]
    
    print(f'{" Start evaluation ":-^40s}\n')
    msg = f'Net: {net.__class__.__name__}\n' + \
          f'Dataset: {dataset.__class__.__name__}\n' + \
          f'Batch size: {batch_size}\n' + \
          f'Device: cuda{str(gpu_ids)}\n'
    print(msg)
    
    torch.cuda.empty_cache()
    
    net = torch.nn.DataParallel(net, device_ids=gpu_ids).cuda()
    
    net.eval()
    torch.set_grad_enabled(False)
    transform.eval()

    # Dictionary containing dataset type-specific subsets and slice indices
    dataset_info = {
        'train': (dataset.train_dataset, dataset.train_case_slice_indices),
        'valid': (dataset.valid_dataset, dataset.valid_case_slice_indices),
        'test': (dataset.test_dataset, dataset.test_case_slice_indices)
    }

    for dataset_type, (subset, case_slice_indices) in dataset_info.items():
        # Call the prediction function for each dataset type with specific data
        create_predict_masks_for_type(net, dataset, case_slice_indices, subset, transform,
                                      data_path, output_path, batch_size, num_workers, dataset_type)


def create_predict_masks_for_type(net, dataset, case_slice_indices, subset, transform, data_path, output_path,
                                  batch_size, num_workers, type):
    sampler = SequentialSampler(subset)
    data_loader = DataLoader(subset, batch_size=batch_size, sampler=sampler,
                             num_workers=num_workers, pin_memory=True)

    case = 0
    vol_output = []

    with tqdm(total=len(case_slice_indices) - 1, ascii=True, desc=f'eval/{type}', dynamic_ncols=True) as pbar:
        for batch_idx, data in enumerate(data_loader):
            imgs, idx = data['image'].cuda(), data['index']

            outputs = net(imgs)
            predicts = outputs['output']
            predicts = predicts.argmax(dim=1)

            predicts = predicts.cpu().detach().numpy()
            idx = idx.numpy()

            vol_output.append(predicts)

            while case < len(case_slice_indices) - 1 and idx[-1] >= case_slice_indices[case + 1] - 1:
                vol_output = np.concatenate(vol_output, axis=0)
                vol_num_slice = case_slice_indices[case + 1] - case_slice_indices[case]

                vol = vol_output[:vol_num_slice]
                
                # Apply ROI transform only for dense_unet
                if transform.use_roi:
                    roi = dataset.get_roi(case, type=type)
                    print(f"ROI information for case {case}: {roi}")
                    # vol = reverse_transform(vol, roi, dataset, transform)
                else:
                    # For other network types, just ensure correct shape
                    vol = vol.astype(np.uint8)

                case_id = dataset.case_idx_to_case_id(case, type=type)
                affine = np.load(data_path / f'case_{case_id:05d}' / 'affine.npy')
                vol_nii = nib.Nifti1Image(vol, affine)
                vol_nii_filename = output_path / f'case_{case_id:05d}' / f'prediction_{case_id:05d}.nii.gz'
                # Create the directory if it doesn't exist
                vol_nii_filename.parent.mkdir(parents=True, exist_ok=True)
                vol_nii.to_filename(str(vol_nii_filename))

                vol_output = [vol_output[vol_num_slice:]]
                case += 1
                pbar.update(1)


def reverse_transform(vol, roi, dataset, transform):
    min_x = max(0, roi['kidney']['min_x'] - transform.roi_error_range)
    max_x = min(vol.shape[-1], roi['kidney']['max_x'] + transform.roi_error_range)
    min_y = max(0, roi['kidney']['min_y'] - transform.roi_error_range)
    max_y = min(vol.shape[-2], roi['kidney']['max_y'] + transform.roi_error_range)
    min_z = max(0, roi['kidney']['min_z'] - dataset.roi_error_range)
    max_z = min(roi['vol']['total_z'], roi['kidney']['max_z'] + dataset.roi_error_range)
    
    min_height = roi['vol']['total_y']
    min_width = roi['vol']['total_x']
    
    roi_rows = max_y - min_y
    roi_cols = max_x - min_x
    max_size = max(transform.output_size[0], transform.output_size[1])
    scale = max_size / float(max(roi_cols, roi_rows))
    rows = int(roi_rows * scale)
    cols = int(roi_cols * scale)
    
    if rows < min_height:
        h_pad_top = int((min_height - rows) / 2.0)
        h_pad_bottom = rows + h_pad_top
    else:
        h_pad_top = 0
        h_pad_bottom = min_height
    
    if cols < min_width:
        w_pad_left = int((min_width - cols) / 2.0)
        w_pad_right = cols + w_pad_left
    else:
        w_pad_left = 0
        w_pad_right = min_width
    
    for i in range(len(vol)):
        img = vol[i]
        reverse_padding_img = img[h_pad_top:h_pad_bottom, w_pad_left:w_pad_right]
        reverse_padding_img = reverse_padding_img.astype(np.uint8)
        reverse_resize_img = cv2.resize(reverse_padding_img, dsize=(max_x - min_x, max_y - min_y),
                                        interpolation=cv2.INTER_LINEAR)
        reverse_resize_img = reverse_resize_img.astype(np.int64)
        reverse_img = np.zeros((min_height, min_width))
        reverse_img[min_y:max_y, min_x: max_x] = reverse_resize_img
        vol[i] = reverse_img
    
    size = (1, min_height, min_width)
    vol_min_z = [np.zeros(size) for _ in range(0, min_z)]
    vol_max_z = [np.zeros(size) for _ in range(max_z, roi['vol']['total_z'])]
    
    vol = vol_min_z + [vol] + vol_max_z
    vol = np.concatenate(vol, axis=0)
    
    assert vol.shape == (roi['vol']['total_z'], roi['vol']['total_y'], roi['vol']['total_x'])
    
    return vol


if __name__ == '__main__':
    main()
