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

# --- For NIQE ---
import scipy
import scipy.ndimage
import scipy.io
import scipy.special
import scipy.linalg
import math
from os.path import dirname, join
from scipy.stats import kurtosis

# -------------------------
# BRISQUE
# -------------------------
def calculate_brisque(img: np.ndarray) -> float:
    """
    Calculate BRISQUE score for a given image.
    Source: https://pypi.org/project/brisque/
    
    Parameters:
        img (np.ndarray): Input image (grayscale or color).
    
    Returns:
        float: BRISQUE score.
    """
    from brisque import BRISQUE

    # Convert to grayscale if necessary
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img

    brisque_obj = BRISQUE(url=False)
    score = brisque_obj.score(img=img_gray)
    return score

# -------------------------
# NIQE and supporting functions
# -------------------------

# Precompute values for the AGGD feature extraction
gamma_range = np.arange(0.2, 10, 0.001)
a = scipy.special.gamma(2.0 / gamma_range)
a *= a
b = scipy.special.gamma(1.0 / gamma_range)
c = scipy.special.gamma(3.0 / gamma_range)
prec_gammas = a / (b * c)

def aggd_features(imdata):
    """
    Compute Asymmetric Generalized Gaussian Distribution (AGGD) features.
    """
    imdata = imdata.flatten()
    imdata2 = imdata * imdata
    left_data = imdata2[imdata < 0]
    right_data = imdata2[imdata >= 0]
    left_mean_sqrt = np.sqrt(np.average(left_data)) if left_data.size > 0 else 0
    right_mean_sqrt = np.sqrt(np.average(right_data)) if right_data.size > 0 else 0
    gamma_hat = left_mean_sqrt / right_mean_sqrt if right_mean_sqrt != 0 else np.inf
    imdata2_mean = np.mean(imdata2)
    r_hat = (np.mean(np.abs(imdata)) ** 2) / (np.mean(imdata2)) if imdata2_mean != 0 else np.inf
    rhat_norm = r_hat * (((gamma_hat ** 3 + 1) * (gamma_hat + 1)) / ((gamma_hat ** 2 + 1) ** 2))
    pos = np.argmin((prec_gammas - rhat_norm) ** 2)
    alpha = gamma_range[pos]
    gam1 = scipy.special.gamma(1.0 / alpha)
    gam2 = scipy.special.gamma(2.0 / alpha)
    gam3 = scipy.special.gamma(3.0 / alpha)
    aggdratio = np.sqrt(gam1) / np.sqrt(gam3)
    bl = aggdratio * left_mean_sqrt
    br = aggdratio * right_mean_sqrt
    N = (br - bl) * (gam2 / gam1)
    return (alpha, N, bl, br, left_mean_sqrt, right_mean_sqrt)

def ggd_features(imdata):
    """
    Compute Generalized Gaussian Distribution (GGD) features.
    """
    nr_gam = 1 / prec_gammas
    sigma_sq = np.var(imdata)
    E = np.mean(np.abs(imdata))
    rho = sigma_sq / (E ** 2)
    pos = np.argmin(np.abs(nr_gam - rho))
    return gamma_range[pos], sigma_sq

def paired_product(new_im):
    """
    Compute paired products for NIQE feature extraction.
    """
    shift1 = np.roll(new_im.copy(), 1, axis=1)
    shift2 = np.roll(new_im.copy(), 1, axis=0)
    shift3 = np.roll(np.roll(new_im.copy(), 1, axis=0), 1, axis=1)
    shift4 = np.roll(np.roll(new_im.copy(), 1, axis=0), -1, axis=1)
    H_img = shift1 * new_im
    V_img = shift2 * new_im
    D1_img = shift3 * new_im
    D2_img = shift4 * new_im
    return (H_img, V_img, D1_img, D2_img)

def gen_gauss_window(lw, sigma):
    """
    Generate a Gaussian window.
    """
    sd = np.float32(sigma)
    lw = int(lw)
    weights = [0.0] * (2 * lw + 1)
    weights[lw] = 1.0
    sum_val = 1.0
    sd = sd * sd
    for ii in range(1, lw + 1):
        tmp = np.exp(-0.5 * (ii * ii) / sd)
        weights[lw + ii] = tmp
        weights[lw - ii] = tmp
        sum_val += 2.0 * tmp
    weights = [w / sum_val for w in weights]
    return weights

def compute_image_mscn_transform(image, C=1, avg_window=None, extend_mode='constant'):
    """
    Compute Mean Subtracted Contrast Normalized (MSCN) coefficients.
    """
    if avg_window is None:
        avg_window = gen_gauss_window(3, 7.0 / 6.0)
    assert image.ndim == 2, "Input image must be 2D."
    h, w = image.shape
    mu_image = np.zeros((h, w), dtype=np.float32)
    var_image = np.zeros((h, w), dtype=np.float32)
    image = image.astype('float32')
    scipy.ndimage.correlate1d(image, avg_window, axis=0, output=mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(mu_image, avg_window, axis=1, output=mu_image, mode=extend_mode)
    scipy.ndimage.correlate1d(image ** 2, avg_window, axis=0, output=var_image, mode=extend_mode)
    scipy.ndimage.correlate1d(var_image, avg_window, axis=1, output=var_image, mode=extend_mode)
    var_image = np.sqrt(np.abs(var_image - mu_image ** 2))
    return ((image - mu_image) / (var_image + C), var_image, mu_image)

def _niqe_extract_subband_feats(mscncoefs):
    """
    Extract subband features for NIQE.
    """
    alpha_m, _, _, _, _, _ = aggd_features(mscncoefs.copy())
    pps1, pps2, pps3, pps4 = paired_product(mscncoefs)
    alpha1, N1, bl1, br1, _, _ = aggd_features(pps1)
    alpha2, N2, bl2, br2, _, _ = aggd_features(pps2)
    alpha3, N3, bl3, br3, _, _ = aggd_features(pps3)
    alpha4, N4, bl4, br4, _, _ = aggd_features(pps4)
    feats = np.array([alpha_m, (bl1 + br1) / 2.0,
                      alpha1, N1, bl1, br1,
                      alpha2, N2, bl2, br2,
                      alpha3, N3, bl3, br3,
                      alpha4, N4, bl4, br4])
    return feats

def get_patches_train_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 1, stride)

def get_patches_test_features(img, patch_size, stride=8):
    return _get_patches_generic(img, patch_size, 0, stride)

def extract_on_patches(img, patch_size):
    """
    Extract features on non-overlapping patches.
    """
    h, w = img.shape
    patch_size = int(patch_size)
    patches = []
    for j in range(0, h - patch_size + 1, patch_size):
        for i in range(0, w - patch_size + 1, patch_size):
            patch = img[j:j + patch_size, i:i + patch_size]
            patches.append(patch)
    patches = np.array(patches)
    patch_features = []
    for p in patches:
        patch_features.append(_niqe_extract_subband_feats(p))
    patch_features = np.array(patch_features)
    return patch_features

def _get_patches_generic(img, patch_size, is_train, stride):
    """
    Generic patch extraction for NIQE feature computation.
    """
    h, w = img.shape
    if h < patch_size or w < patch_size:
        print("Input image is too small")
        exit(0)
    hoffset = h % patch_size
    woffset = w % patch_size
    if hoffset > 0:
        img = img[:-hoffset, :]
    if woffset > 0:
        img = img[:, :-woffset]
    img = img.astype(np.float32)
    # Resize image to half size for second-level features
    img2 = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_CUBIC)
    mscn1, _, _ = compute_image_mscn_transform(img)
    mscn1 = mscn1.astype(np.float32)
    mscn2, _, _ = compute_image_mscn_transform(img2)
    mscn2 = mscn2.astype(np.float32)
    feats_lvl1 = extract_on_patches(mscn1, patch_size)
    feats_lvl2 = extract_on_patches(mscn2, patch_size / 2)
    feats = np.hstack((feats_lvl1, feats_lvl2))
    return feats

def niqe(inputImgData):
    """
    Compute the NIQE score for a given 2D image.
    Source: https://github.com/guptapraful/niqe
    
    Parameters:
        inputImgData (np.ndarray): Grayscale image.
    
    Returns:
        float: NIQE score.
    """
    patch_size = 96
    module_path = dirname(__file__)
    # Load pre-trained NIQE parameters from a .mat file
    params = scipy.io.loadmat(join(module_path, 'data', 'niqe_image_params.mat'))
    pop_mu = np.ravel(params["pop_mu"])
    pop_cov = params["pop_cov"]

    M, N = inputImgData.shape
    assert M > (patch_size * 2 + 1), "Image too small for NIQE computation"
    assert N > (patch_size * 2 + 1), "Image too small for NIQE computation"

    feats = get_patches_test_features(inputImgData, patch_size)
    sample_mu = np.mean(feats, axis=0)
    sample_cov = np.cov(feats.T)
    X = sample_mu - pop_mu
    covmat = ((pop_cov + sample_cov) / 2.0)
    pinvmat = scipy.linalg.pinv(covmat)
    niqe_score = np.sqrt(np.dot(np.dot(X, pinvmat), X))
    return niqe_score

def calculate_niqe(img: np.ndarray) -> float:
    """
    Wrapper function to calculate NIQE score.
    Ensures the image is in grayscale.
    
    Parameters:
        img (np.ndarray): Input image.
    
    Returns:
        float: NIQE score.
    """
    if img.ndim == 3:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img
    return niqe(img_gray)

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