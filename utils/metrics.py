import os
import math
import numpy as np
import cv2
import PIL as Image
import torch
from torchvision.utils import make_grid
from torchmetrics.image import VisualInformationFidelity, SpatialCorrelationCoefficient
import torch.nn.functional as F

def tensor2img(tensor, out_type=np.uint8, min_max=(-1, 1)):
    '''
    Converts a torch Tensor into an image Numpy array
    Input: 4D(B,(3/1),H,W), 3D(C,H,W), or 2D(H,W), any range, RGB channel order
    Output: 3D(H,W,C) or 2D(H,W), [0,255], np.uint8 (default)
    '''
    #tensor = tensor.squeeze().float().cpu().clamp_(*min_max)  # clamp
    tensor = tensor.squeeze(0).float().cpu().clamp_(*min_max)  # clamp
    tensor = (tensor - min_max[0]) / \
        (min_max[1] - min_max[0])  # to range [0,1]
    n_dim = tensor.dim()
    if n_dim == 4:
        n_img = len(tensor)
        img_np = make_grid(tensor, nrow=int(
            math.sqrt(n_img)), normalize=False).numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 3:
        img_np = tensor.numpy()
        img_np = np.transpose(img_np, (1, 2, 0))  # HWC, RGB
    elif n_dim == 2:
        img_np = tensor.numpy()
    else:
        raise TypeError(
            'Only support 4D, 3D and 2D tensor. But received with dimension: {:d}'.format(n_dim))
    if out_type == np.uint8:
        img_np = (img_np * 255.0).round()
        # Important. Unlike matlab, numpy.unit8() WILL NOT round by default.
    return img_np.astype(out_type)


def grad(img):
    sobel_x = torch.tensor([[-1, 0, 1],
                            [-2, 0, 2],
                            [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[-1, -2, -1],
                            [0, 0, 0],
                            [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)
    grad_x = F.conv2d(img, sobel_x, padding=1)
    grad_y = F.conv2d(img, sobel_y, padding=1)
    magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)
    direction = torch.atan2(grad_y, grad_x)

    return grad_x, grad_y, magnitude, direction


import cv2
def mergy_Y_RGB_to_YCbCr(img1, img2):
    Y_channel = img1[:, :, 0]

    img2_YCbCr = cv2.cvtColor(img2, cv2.COLOR_RGB2YCrCb)
    CbCr_channels = img2_YCbCr[:, :, 1:]
    merged_img_YCbCr = np.concatenate((Y_channel[:, :, np.newaxis], CbCr_channels), axis=2)
    merged_img = cv2.cvtColor(merged_img_YCbCr, cv2.COLOR_YCrCb2RGB)

    return merged_img

def save_img(img, img_path, mode='RGB'):
    h,w,c = img.shape
    if c==3:
        cv2.imwrite(img_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(img_path, img)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))


def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()


def calculate_ssim(img1, img2):
    '''calculate SSIM
    the same outputs as MATLAB's
    img1, img2: [0, 255]
    '''
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if img1.ndim == 2:
        return ssim(img1, img2)
    elif img1.ndim == 3:
        if img1.shape[2] == 3:
            ssims = []
            for i in range(3):
                ssims.append(ssim(img1, img2))
            return np.array(ssims).mean()
        elif img1.shape[2] == 1:
            return ssim(np.squeeze(img1), np.squeeze(img2))
    else:
        raise ValueError('Wrong input image dimensions.')

def calculate_mi(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    hist_associate = torch.zeros(256,256).cuda()

    for i in range(img1.shape[0]):
        for j in range(img1.shape[1]):
            hist_associate[int(img1[i,j]),int(img2[i,j])] += 1

    hist_associate /= hist_associate.sum()

    # Calculate MI
    p1 = np.sum(hist_associate, axis=1)
    p2 = np.sum(hist_associate, axis=0)

    mi = 0
    for i in range(256):
        for j in range(256):
            if hist_associate[i,j] > 0:
                mi += hist_associate[i,j] * torch.log(hist_associate[i,j] / (p1[i] * p2[j]))

    return mi.item()

def calculate_vif(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    # Convert images to float32
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # Calculate VIF
    vif = VisualInformationFidelity()
    vif_score = vif(img1, img2).item()
    return vif_score

def calculate_qabf(img1, img2, img_fusion, p:int):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')
    if not img1.shape == img_fusion.shape:
        raise ValueError('Fusion image must have the same dimensions as input images.')

    # Convert images to float32
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    img_fusion = img_fusion.astype(np.float32)

    _, _, grad_mag1, _ = grad(img1)
    _, _, grad_mag2, _ = grad(img2)
    _, _, grad_magf, _ = grad(img_fusion)

    qaf = (2 * grad_mag1 * grad_magf + 1e-6) / (grad_mag1 ** 2 + grad_magf ** 2 + 1e-6)
    qbf = (2 * grad_mag2 * grad_magf + 1e-6) / (grad_mag2 ** 2 + grad_magf ** 2 + 1e-6)

    wa = grad_mag1 ** p
    wb = grad_mag2 ** p

    qabf = (torch.sum(qaf * wa) + torch.sum(qbf * wb)) / (torch.sum(wa) + torch.sum(wb))
    return qabf

def calculate_scc(img1, img2):
    if not img1.shape == img2.shape:
        raise ValueError('Input images must have the same dimensions.')

    # Convert images to float32
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # Calculate VIF
    scc = SpatialCorrelationCoefficient()
    scc_score = scc(img1, img2)
    return scc_score