# -*- coding: utf-8 -*-
import glob, sys
import os, losses, utils
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
from natsort import natsorted
from models.DHIT_Net import CONFIGS as CONFIGS_TM
import models.DHIT_Net as DHIT_Net
from scipy.ndimage import distance_transform_edt
import torch.nn.functional as F

# Define 35 VOI labels (labels 1-35)
VOI_lbls = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]

def compute_surface_distances(pred, gt, spacing=(1.0, 1.0, 1.0)):
    """
    Calculate surface distances between two binary masks
    """
    pred_border = np.logical_xor(pred, np.logical_and(
        np.logical_and(
            np.pad(pred, ((1,1),(1,1),(1,1)), mode='constant')[2:, 1:-1, 1:-1],
            np.pad(pred, ((1,1),(1,1),(1,1)), mode='constant')[:-2, 1:-1, 1:-1]
        ),
        np.logical_and(
            np.pad(pred, ((1,1),(1,1),(1,1)), mode='constant')[1:-1, 2:, 1:-1],
            np.pad(pred, ((1,1),(1,1),(1,1)), mode='constant')[1:-1, :-2, 1:-1]
        )
    ))
    pred_border = np.logical_or(pred_border, np.logical_xor(pred, np.logical_and(
        np.pad(pred, ((1,1),(1,1),(1,1)), mode='constant')[1:-1, 1:-1, 2:],
        np.pad(pred, ((1,1),(1,1),(1,1)), mode='constant')[1:-1, 1:-1, :-2]
    )))

    gt_border = np.logical_xor(gt, np.logical_and(
        np.logical_and(
            np.pad(gt, ((1,1),(1,1),(1,1)), mode='constant')[2:, 1:-1, 1:-1],
            np.pad(gt, ((1,1),(1,1),(1,1)), mode='constant')[:-2, 1:-1, 1:-1]
        ),
        np.logical_and(
            np.pad(gt, ((1,1),(1,1),(1,1)), mode='constant')[1:-1, 2:, 1:-1],
            np.pad(gt, ((1,1),(1,1),(1,1)), mode='constant')[1:-1, :-2, 1:-1]
        )
    ))
    gt_border = np.logical_or(gt_border, np.logical_xor(gt, np.logical_and(
        np.pad(gt, ((1,1),(1,1),(1,1)), mode='constant')[1:-1, 1:-1, 2:],
        np.pad(gt, ((1,1),(1,1),(1,1)), mode='constant')[1:-1, 1:-1, :-2]
    )))

    dt_pred = distance_transform_edt(~pred_border, sampling=spacing)
    dt_gt = distance_transform_edt(~gt_border, sampling=spacing)

    pred_surface_pts = np.where(pred_border)
    gt_surface_pts = np.where(gt_border)

    if len(pred_surface_pts[0]) == 0 or len(gt_surface_pts[0]) == 0:
        return None, None

    distances_pred_to_gt = dt_gt[pred_surface_pts]
    distances_gt_to_pred = dt_pred[gt_surface_pts]

    return distances_pred_to_gt, distances_gt_to_pred

def compute_hd95(pred, gt, spacing=(1.0, 1.0, 1.0)):
    """
    Calculate 95th percentile Hausdorff Distance
    """
    distances_pred_to_gt, distances_gt_to_pred = compute_surface_distances(pred, gt, spacing)

    if distances_pred_to_gt is None or distances_gt_to_pred is None:
        return np.nan

    hd95_pred_to_gt = np.percentile(distances_pred_to_gt, 95)
    hd95_gt_to_pred = np.percentile(distances_gt_to_pred, 95)

    return max(hd95_pred_to_gt, hd95_gt_to_pred)

def compute_assd(pred, gt, spacing=(1.0, 1.0, 1.0)):
    """
    Calculate Average Symmetric Surface Distance
    """
    distances_pred_to_gt, distances_gt_to_pred = compute_surface_distances(pred, gt, spacing)

    if distances_pred_to_gt is None or distances_gt_to_pred is None:
        return np.nan

    return (np.mean(distances_pred_to_gt) + np.mean(distances_gt_to_pred)) / 2.0

def dice_val_voi_per_structure(y_pred, y_true, voi_labels):
    """
    Calculate DSC for each VOI structure separately
    Returns a list of DSC values, one per structure
    """
    pred = y_pred.detach().cpu().numpy()[0, 0, ...]
    true = y_true.detach().cpu().numpy()[0, 0, ...]
    DSCs = []
    for i in voi_labels:
        pred_i = pred == i
        true_i = true == i
        intersection = np.sum(pred_i * true_i)
        union = np.sum(pred_i) + np.sum(true_i)
        dsc = (2. * intersection) / (union + 1e-5)
        DSCs.append(dsc)
    return DSCs

def compute_hd95_per_structure(pred_seg, gt_seg, voi_labels, spacing=(1.0, 1.0, 1.0)):
    """
    Calculate HD95 for each VOI structure separately
    Returns a list of HD95 values, one per structure
    """
    hd95_list = []
    for label in voi_labels:
        pred_i = (pred_seg == label).astype(np.uint8)
        gt_i = (gt_seg == label).astype(np.uint8)
        if pred_i.sum() == 0 and gt_i.sum() == 0:
            hd95_list.append(np.nan)
        elif pred_i.sum() == 0 or gt_i.sum() == 0:
            hd95_list.append(np.nan)
        else:
            hd95_list.append(compute_hd95(pred_i, gt_i, spacing))
    return hd95_list

def compute_assd_per_structure(pred_seg, gt_seg, voi_labels, spacing=(1.0, 1.0, 1.0)):
    """
    Calculate ASSD for each VOI structure separately
    Returns a list of ASSD values, one per structure
    """
    assd_list = []
    for label in voi_labels:
        pred_i = (pred_seg == label).astype(np.uint8)
        gt_i = (gt_seg == label).astype(np.uint8)
        if pred_i.sum() == 0 and gt_i.sum() == 0:
            assd_list.append(np.nan)
        elif pred_i.sum() == 0 or gt_i.sum() == 0:
            assd_list.append(np.nan)
        else:
            assd_list.append(compute_assd(pred_i, gt_i, spacing))
    return assd_list

def jacobian_determinant_gpu(disp):
    """
    GPU version of Jacobian determinant calculation
    disp: [B, 3, H, W, D] torch tensor on GPU
    """
    B, C, H, W, D = disp.shape

    gradx_kernel = torch.tensor([-0.5, 0, 0.5], device=disp.device).reshape(1, 1, 3, 1, 1)
    grady_kernel = torch.tensor([-0.5, 0, 0.5], device=disp.device).reshape(1, 1, 1, 3, 1)
    gradz_kernel = torch.tensor([-0.5, 0, 0.5], device=disp.device).reshape(1, 1, 1, 1, 3)

    gradx_disp = torch.stack([
        F.conv3d(disp[:, 0:1, :, :, :], gradx_kernel, padding=(1, 0, 0)),
        F.conv3d(disp[:, 1:2, :, :, :], gradx_kernel, padding=(1, 0, 0)),
        F.conv3d(disp[:, 2:3, :, :, :], gradx_kernel, padding=(1, 0, 0))
    ], dim=1).squeeze(2)

    grady_disp = torch.stack([
        F.conv3d(disp[:, 0:1, :, :, :], grady_kernel, padding=(0, 1, 0)),
        F.conv3d(disp[:, 1:2, :, :, :], grady_kernel, padding=(0, 1, 0)),
        F.conv3d(disp[:, 2:3, :, :, :], grady_kernel, padding=(0, 1, 0))
    ], dim=1).squeeze(2)

    gradz_disp = torch.stack([
        F.conv3d(disp[:, 0:1, :, :, :], gradz_kernel, padding=(0, 0, 1)),
        F.conv3d(disp[:, 1:2, :, :, :], gradz_kernel, padding=(0, 0, 1)),
        F.conv3d(disp[:, 2:3, :, :, :], gradz_kernel, padding=(0, 0, 1))
    ], dim=1).squeeze(2)

    grad_disp = torch.cat([gradx_disp, grady_disp, gradz_disp], dim=1)[0]

    jacobian = grad_disp.reshape(3, 3, H, W, D) + torch.eye(3, device=disp.device).reshape(3, 3, 1, 1, 1)
    jacobian = jacobian[:, :, 2:-2, 2:-2, 2:-2]

    jacdet = (jacobian[0, 0, :, :, :] * (jacobian[1, 1, :, :, :] * jacobian[2, 2, :, :, :] -
                                          jacobian[1, 2, :, :, :] * jacobian[2, 1, :, :, :]) -
              jacobian[1, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[2, 2, :, :, :] -
                                          jacobian[0, 2, :, :, :] * jacobian[2, 1, :, :, :]) +
              jacobian[2, 0, :, :, :] * (jacobian[0, 1, :, :, :] * jacobian[1, 2, :, :, :] -
                                          jacobian[0, 2, :, :, :] * jacobian[1, 1, :, :, :]))

    return jacdet

def main():
    test_dir = r'/path/to/dataset/test/'
    model_idx = -1
    weights = [1, 1, 1]
    model_folder = 'DHIT_Net_ncc_{}_diffusion_{}/'.format(weights[0], weights[1])
    model_dir = 'experiments/' + model_folder

    config = CONFIGS_TM['DHIT_Net']
    model = DHIT_Net.DHIT_Net(config)
    best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[model_idx])['state_dict']
    print('Best model: {}'.format(natsorted(os.listdir(model_dir))[model_idx]))
    model.load_state_dict(best_model)
    model.cuda()

    img_size = (160, 192, 224)
    reg_model = utils.register_model(img_size, 'nearest')
    reg_model.cuda()

    test_composed = transforms.Compose([trans.NumpyType((np.float32, np.int16)),])
    test_files = glob.glob(test_dir + '*.pkl')
    test_set = datasets.BrainInferDataset(test_files, transforms=test_composed)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, drop_last=True)

    print('='*60)
    print('Starting evaluation on 35 VOI structures')
    print('VOI labels: {}'.format(VOI_lbls))
    print('='*60)

    # Store raw values: (num_structures x num_samples) matrix
    num_samples = len(test_files)
    num_voi = len(VOI_lbls)

    all_dsc    = np.zeros((num_voi, num_samples))         # (35 x N)
    all_hd95   = np.full((num_voi, num_samples), np.nan)  # (35 x N)
    all_assd   = np.full((num_voi, num_samples), np.nan)  # (35 x N)
    all_sdlogj = np.zeros(num_samples)                    # (N,)

    with torch.no_grad():
        for stdy_idx, data in enumerate(test_loader):
            # Extract sample name from file path
            current_file_path = test_files[stdy_idx]
            sample_name = os.path.basename(current_file_path).replace('.pkl', '')

            model.eval()
            data = [t.cuda() for t in data]
            x, y, x_seg, y_seg = data[0], data[1], data[2], data[3]

            x_in = torch.cat((x, y), dim=1)
            x_def, flow = model(x_in)

            def_out = reg_model([x_seg.cuda().float(), flow.cuda()])

            # SDlogJ using GPU
            jac_det = jacobian_determinant_gpu(flow)
            jac_det_clipped = torch.clamp(jac_det + 3, min=1e-9, max=1e9)
            log_jac_det = torch.log(jac_det_clipped)
            sdlogj = log_jac_det.std().item()
            all_sdlogj[stdy_idx] = sdlogj

            # DSC per structure (35 values)
            dsc_per_struct = dice_val_voi_per_structure(def_out.long(), y_seg.long(), VOI_lbls)
            all_dsc[:, stdy_idx] = dsc_per_struct

            # HD95 and ASSD per structure (35 values each)
            def_out_np = def_out.detach().cpu().numpy()[0, 0]
            y_seg_np = y_seg.detach().cpu().numpy()[0, 0]
            hd95_per_struct = compute_hd95_per_structure(def_out_np, y_seg_np, VOI_lbls)
            assd_per_struct = compute_assd_per_structure(def_out_np, y_seg_np, VOI_lbls)
            all_hd95[:, stdy_idx] = hd95_per_struct
            all_assd[:, stdy_idx] = assd_per_struct

            # Print progress
            print('[{}/{}] Sample: {} | DSC: {:.4f} | HD95: {:.4f} | ASSD: {:.4f} | SDlogJ: {:.4f}'.format(
                stdy_idx+1, num_samples, sample_name,
                np.mean(dsc_per_struct),
                np.nanmean(hd95_per_struct),
                np.nanmean(assd_per_struct),
                sdlogj))

    # Final statistics
    print('='*60)
    print('FINAL STATISTICS (35 VOI structures):')
    print('='*60)
    print('DSC:    {:.3f} +/- {:.3f}'.format(
        all_dsc.mean(),
        all_dsc.std()))
    print('HD95:   {:.3f} +/- {:.3f}'.format(
        np.nanmean(all_hd95),
        np.nanstd(all_hd95)))
    print('ASSD:   {:.3f} +/- {:.3f}'.format(
        np.nanmean(all_assd),
        np.nanstd(all_assd)))
    print('SDlogJ: {:.4f} +/- {:.4f}'.format(
        all_sdlogj.mean(),
        all_sdlogj.std()))
    print('='*60)

if __name__ == '__main__':
    GPU_iden = 0
    GPU_num = torch.cuda.device_count()
    print('Number of GPU: ' + str(GPU_num))
    for GPU_idx in range(GPU_num):
        GPU_name = torch.cuda.get_device_name(GPU_idx)
        print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
    torch.cuda.set_device(GPU_iden)
    GPU_avai = torch.cuda.is_available()
    print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
    print('If the GPU is available? ' + str(GPU_avai))
    main()