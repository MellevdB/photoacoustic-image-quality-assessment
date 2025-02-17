"""
nr_metrics.py

No-Reference Image Quality Metrics Module

This module implements several no-reference (NR) image quality metrics.

Implemented:
    - BRISQUE (using the 'brisque' package)
    - NIQE (based on the implementation from https://github.com/guptapraful/niqe)
    - NIQE-K (a modified version of NIQE, based on Outtas et al., ISIVC 2016)

Placeholders (to be implemented later):
    - DIIVINE (Moorthy and Bovik, TIP 2011)
    - CBIQ (Ye and Doermann, TIP 2012)
    - LBIQ (Tang, Joshi and Kapoor, CVPR 2011)
    - ILNIQE (Zhang, Zhang and Bovik, TIP 2015)
    - BLINDS (Saad, Bovik and Charrier, TIP 2012)
    - BIQES (Saha and Wu, TIP 2015)
    - MSM (DL-based from Yuan et al., 2024)
"""

import numpy as np
import cv2
from PIL import Image
from imquality.brisque import score

# --- For NIQE ---
import scipy
import scipy.ndimage
import scipy.io
import scipy.special
import scipy.linalg
import math
from os.path import dirname, join
from scipy.stats import kurtosis
from scipy.ndimage.filters import convolve
from scipy.special import gamma
from brisque import BRISQUE

# -------------------------
# BRISQUE
# https://github.com/ocampor/image-quality.git
# -------------------------

MODEL_PATH = "models/brisque"

def calculate_brisque(img: np.ndarray) -> float:
    """
    Calculate BRISQUE score using the imquality-based implementation.
    """
    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    # # Debug: Check for NaN or empty images
    # print(f"BRISQUE input shape: {img.size}")
    # np_img = np.asarray(img)
    # print(f"Min: {np.min(np_img)}, Max: {np.max(np_img)}, Mean: {np.mean(np_img)}")

    try:
        brisque_score = score(img)
        return brisque_score
    except Exception as e:
        print(f"BRISQUE computation failed: {e}")
        return float('nan')
# -------------------------
# NIQE and supporting functions
# -------------------------

def estimate_aggd_param(block):
    """Estimate AGGD (Asymmetric Generalized Gaussian Distribution) parameters.
    
    Args:
        block (ndarray): 2D image block.
    
    Returns:
        tuple: (alpha, beta_l, beta_r)
    """
    block = block.flatten()
    gam = np.arange(0.2, 10.001, 0.001)  # length ~9801
    gam_reciprocal = 1.0 / gam
    r_gam = np.square(gamma(gam_reciprocal * 2)) / (gamma(gam_reciprocal) * gamma(gam_reciprocal * 3))
    
    left_std = np.sqrt(np.mean(block[block < 0] ** 2))
    right_std = np.sqrt(np.mean(block[block > 0] ** 2))
    gammahat = left_std / right_std if right_std != 0 else np.inf
    rhat = (np.mean(np.abs(block))) ** 2 / np.mean(block ** 2)
    rhatnorm = (rhat * (gammahat ** 3 + 1) * (gammahat + 1)) / ((gammahat ** 2 + 1) ** 2)
    array_position = np.argmin((r_gam - rhatnorm) ** 2)
    
    alpha = gam[array_position]
    beta_l = left_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    beta_r = right_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    return (alpha, beta_l, beta_r)

def compute_feature(block):
    """Compute features for a given block.
    
    Args:
        block (ndarray): 2D image block.
    
    Returns:
        list: Features (length 18).
    """
    feat = []
    alpha, beta_l, beta_r = estimate_aggd_param(block)
    feat.extend([alpha, (beta_l + beta_r) / 2])
    
    shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
    for shift in shifts:
        shifted_block = np.roll(block, shift, axis=(0, 1))
        alpha, beta_l, beta_r = estimate_aggd_param(block * shifted_block)
        mean_val = (beta_r - beta_l) * (gamma(2 / alpha) / gamma(1 / alpha))
        feat.extend([alpha, mean_val, beta_l, beta_r])
    return feat

def niqe(img, mu_pris_param, cov_pris_param, gaussian_window, block_size_h=96, block_size_w=96):
    """
    Compute NIQE (Natural Image Quality Evaluator) metric.
    
    Args:
        img (ndarray): Grayscale image with shape (h, w) in range [0, 255].
        mu_pris_param (ndarray): Mean parameter from pristine images.
        cov_pris_param (ndarray): Covariance parameter from pristine images.
        gaussian_window (ndarray): A 7x7 Gaussian window.
        block_size_h (int): Block height (default: 96).
        block_size_w (int): Block width (default: 96).
    
    Returns:
        float: NIQE score.
    """
    assert img.ndim == 2, 'Input image must be grayscale with shape (h, w).'
    h, w = img.shape
    num_block_h = math.floor(h / block_size_h)
    num_block_w = math.floor(w / block_size_w)
    img_cropped = img[:num_block_h * block_size_h, :num_block_w * block_size_w]
    
    distparam = []
    for scale in (1, 2):
        mu = convolve(img_cropped, gaussian_window, mode='nearest')
        sigma = np.sqrt(np.abs(convolve(np.square(img_cropped), gaussian_window, mode='nearest') - np.square(mu)))
        img_normalized = (img_cropped - mu) / (sigma + 1)
        
        feat = []
        for idx_h in range(num_block_h):
            for idx_w in range(num_block_w):
                block = img_normalized[idx_h*block_size_h//scale:(idx_h+1)*block_size_h//scale,
                                        idx_w*block_size_w//scale:(idx_w+1)*block_size_w//scale]
                feat.append(compute_feature(block))
        distparam.append(np.array(feat))
        if scale == 1:
            h2, w2 = img_cropped.shape
            img_cropped = cv2.resize(img_cropped/255., (w2//2, h2//2), interpolation=cv2.INTER_LINEAR)*255.
    
    distparam = np.concatenate(distparam, axis=1)
    mu_distparam = np.nanmean(distparam, axis=0)
    distparam_no_nan = distparam[~np.isnan(distparam).any(axis=1)]
    cov_distparam = np.cov(distparam_no_nan, rowvar=False)
    
    invcov_param = np.linalg.pinv((cov_pris_param + cov_distparam) / 2)
    X = mu_pris_param - mu_distparam
    quality = np.sqrt(np.dot(np.dot(X, invcov_param), X))
    return quality

def calculate_niqe(img: np.ndarray, crop_border=0, input_order='HWC', convert_to='y') -> float:
    """
    Calculate NIQE metric.
    
    Args:
        img (ndarray): Input image with range [0, 255]. Can be in 'HW', 'HWC', or 'CHW' format.
        crop_border (int): Border to crop (default 0).
        input_order (str): 'HW', 'HWC', or 'CHW' (default 'HWC').
        convert_to (str): Convert to 'y' (default) or 'gray'.
    
    Returns:
        float: NIQE score.
    """
    # Load pristine parameters from a .npz file
    niqe_pris_params = np.load('basicsr/metrics/niqe_pris_params.npz')
    mu_pris_param = niqe_pris_params['mu_pris_param']
    cov_pris_param = niqe_pris_params['cov_pris_param']
    gaussian_window = niqe_pris_params['gaussian_window']
    
    img = img.astype(np.float32)
    if input_order != 'HW':
        img = reorder_image(img, input_order=input_order)
        if convert_to == 'y':
            img = to_y_channel(img)
        elif convert_to == 'gray':
            img = cv2.cvtColor(img / 255., cv2.COLOR_BGR2GRAY) * 255.
        img = np.squeeze(img)
    
    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border]
    
    niqe_result = niqe(img, mu_pris_param, cov_pris_param, gaussian_window)
    return niqe_result

# -------------------------
# NIQE-K (Modified NIQE)
# -------------------------
def calculate_niqe_k(img: np.ndarray) -> float:
    """
    Calculate modified NIQE (NIQE-K) score.
    
    NIQE-K = NIQE * (kurtosis(log|FFT(image)|) / σ(log|FFT(image)|))
    
    Source:
        M. Outtas et al., "A study on the usability of opinion-unaware no-reference natural image quality
        metrics in the context of medical images," ISIVC 2016.
    
    Parameters:
        img (np.ndarray): Input image.
    
    Returns:
        float: NIQE-K score.
    """
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    niqe_score = calculate_niqe(img_gray)
    fft_img = np.fft.fft2(img_gray)
    fft_log = np.log(np.abs(fft_img) + 1e-8)
    k = kurtosis(fft_log, fisher=False)
    sigma = np.std(fft_log)
    ratio = k / sigma if sigma != 0 else 0
    return niqe_score * ratio

# -------------------------
# Placeholders for other NR metrics
# -------------------------
def calculate_diivine(img: np.ndarray) -> float:
    """
    Placeholder for DIIVINE metric.
    Source: A. K. Moorthy and A. C. Bovik, "Blind Image Quality Assessment: From Natural Scene Statistics to Perceptual Quality,"
            IEEE Transactions on Image Processing, 2011.
    """
    # To be implemented
    return None

def calculate_cbiq(img: np.ndarray) -> float:
    """
    Placeholder for CBIQ metric.
    Source: P. Ye and D. Doermann, "No-Reference Image Quality Assessment Using Visual Codebooks,"
            IEEE Transactions on Image Processing, 2012.
    """
    # To be implemented
    return None

def calculate_lbiq(img: np.ndarray) -> float:
    """
    Placeholder for LBIQ metric.
    Source: H. Tang, N. Joshi and A. Kapoor, "Learning a blind measure of perceptual image quality,"
            CVPR 2011.
    """
    # To be implemented
    return None

def calculate_ilniqe(img: np.ndarray) -> float:
    """
    Placeholder for ILNIQE metric.
    Source: L. Zhang, L. Zhang and A. C. Bovik, "A Feature-Enriched Completely Blind Image Quality Evaluator,"
            IEEE Transactions on Image Processing, 2015.
    """
    # To be implemented
    return None

def calculate_blinds(img: np.ndarray) -> float:
    """
    Placeholder for BLINDS metric.
    Source: M. A. Saad, A. C. Bovik and C. Charrier, "Blind Image Quality Assessment: A Natural Scene Statistics Approach in the DCT Domain,"
            IEEE Transactions on Image Processing, 2012.
    """
    # To be implemented
    return None

def calculate_biqes(img: np.ndarray) -> float:
    """
    Placeholder for BIQES metric.
    Source: Saha and Wu, "Utilizing image scales towards totally training free blind image quality assessment,"
            IEEE Transactions on Image Processing, 2015.
    """
    # To be implemented
    return None

def calculate_msm(img: np.ndarray) -> float:
    """
    Placeholder for MSM metric (DL-based).
    Source: Yuan et al., 2024.
    """
    # To be implemented
    return None

# Helper functions
def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.

    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.

    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.

    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(
            f'Wrong input_order {input_order}. Supported input_orders are '
            "'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img


def to_y_channel(img):
    """Change to Y channel of YCbCr.

    Args:
        img (ndarray): Images with range [0, 255].

    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.

def bgr2ycbcr(img, y_only=False):
    """Convert a BGR image to YCbCr image.

    The bgr version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786],
                  [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img

def _convert_input_type_range(img):
    """Convert the type and range of the input image.

    It converts the input image to np.float32 type and range of [0, 1].
    It is mainly used for pre-processing the input image in colorspace
    convertion functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].

    Returns:
        (ndarray): The converted image with type of np.float32 and range of
            [0, 1].
    """
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError('The img type should be np.float32 or np.uint8, '
                        f'but got {img_type}')
    return img


def _convert_output_type_range(img, dst_type):
    """Convert the type and range of the image according to dst_type.

    It converts the image to desired type and range. If `dst_type` is np.uint8,
    images will be converted to np.uint8 type with range [0, 255]. If
    `dst_type` is np.float32, it converts the image to np.float32 type with
    range [0, 1].
    It is mainly used for post-processing images in colorspace convertion
    functions such as rgb2ycbcr and ycbcr2rgb.

    Args:
        img (ndarray): The image to be converted with np.float32 type and
            range [0, 255].
        dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
            converts the image to np.uint8 type with range [0, 255]. If
            dst_type is np.float32, it converts the image to np.float32 type
            with range [0, 1].

    Returns:
        (ndarray): The converted image with desired type and range.
    """
    if dst_type not in (np.uint8, np.float32):
        raise TypeError('The dst_type should be np.float32 or np.uint8, '
                        f'but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)