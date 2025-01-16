import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def compute_psnr(gt, pred):
    return psnr(gt, pred, data_range=gt.max() - gt.min())

def compute_ssim(gt, pred):
    return ssim(gt, pred, data_range=gt.max() - gt.min())
