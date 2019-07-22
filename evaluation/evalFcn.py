import torch
import numpy as np
import math

def getEvalFcn(type):
    if '_' in type:
        params = type.split('_')
        type = params[-1]
        params = [float(param) for param in params[:-1]]
        return lambda gt, output: globals()[type](gt, output, *params)
    else:
        return globals()[type]

# L1 loss between gt and output
def l1(gt, output):
    if len(gt) == 0:
        loss = 0
    else:
        loss = torch.mean(torch.abs(output - gt)).item()  # end-point-error
    return loss


# Compute outlier proportion. Ported from kitti devkit
# http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo
# 'For this benchmark, we consider a pixel to be correctly estimated if the disparity or flow end-point error is <3px or <5%'
def outlier(gt, output, npx=3, acc=0.05):
    dErr = torch.abs(gt - output)
    nTotal = float(torch.numel(gt))
    nWrong = float(torch.sum((dErr > npx) & ((dErr / gt) > acc)).item())
    return nWrong / nTotal * 100

def outlierPSMNet(disp_true, pred_disp):
    disp_true = disp_true.squeeze(1)
    pred_disp = pred_disp.squeeze(1)
    disp_true = disp_true.data.cpu()
    pred_disp = pred_disp.data.cpu()
    # computing 3-px error#
    true_disp = disp_true
    index = np.argwhere(true_disp > 0)
    disp_true[index[0][:], index[1][:], index[2][:]] = np.abs(
        true_disp[index[0][:], index[1][:], index[2][:]] - pred_disp[index[0][:], index[1][:], index[2][:]])
    correct = (disp_true[index[0][:], index[1][:], index[2][:]] < 3) | (
                disp_true[index[0][:], index[1][:], index[2][:]] < true_disp[
            index[0][:], index[1][:], index[2][:]] * 0.05)
    torch.cuda.empty_cache()

    return (1 - (float(torch.sum(correct)) / float(len(index[0])))) * 100

def psnr(sr, hr, scale=2, rgb_range=1, dataset=None):
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    if dataset and dataset.dataset.benchmark:
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)
