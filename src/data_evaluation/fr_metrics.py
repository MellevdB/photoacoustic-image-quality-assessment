import numpy as np
import cv2
from phasepack import phasecong as pc
# from neutompy.metrics.metrics import GMSD
from sewar.full_ref import vifp, uqi, msssim
from skimage.filters import threshold_local
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

def calculate_s3im(org_img: np.ndarray, pred_img: np.ndarray) -> float:
    """
    Sparse SSIM (S3IM) metric for image quality assessment.

    Args:
        org_img (np.ndarray): Reference image.
        pred_img (np.ndarray): Predicted image.

    Returns:
        float: Sparse SSIM value.
    """
    # # Debug: Check for NaN or empty images
    # print(f"S3IM input shapes - org_img: {org_img.shape}, pred_img: {pred_img.shape}")
    # print(f"S3IM input types - org_img: {org_img.dtype}, pred_img: {pred_img.dtype}")
    # print(f"S3IM input ranges - org_img: [{org_img.min()}, {org_img.max()}], pred_img: [{pred_img.min()}, {pred_img.max()}]")

    # Define parameters
    neighborhood_size = (org_img.shape[0] // 16) * 2 + 1  # Adaptive window size
    sensitivity = 0.5  # Can be tuned based on dataset noise

    # Adaptive thresholding to create masks
    level1 = threshold_local(org_img, neighborhood_size, method='gaussian', offset=sensitivity)
    level2 = threshold_local(pred_img, neighborhood_size, method='gaussian', offset=sensitivity)

    mask1 = org_img > level1
    mask2 = pred_img > level2
    mask = np.logical_or(mask1, mask2)  # Combine masks

    # Apply mask to images
    masked_img1 = org_img * mask
    masked_img2 = pred_img * mask

    # Compute SSIM only in masked regions
    ssim_map = structural_similarity(masked_img1, masked_img2, data_range=org_img.max() - org_img.min(), full=True)[1]
    s3im_map = ssim_map * mask.astype(np.float64)

    # Compute Sparse SSIM
    if np.any(mask):
        s3im_val = np.sum(s3im_map) / np.sum(mask)
    else:
        s3im_val = 0  # Safe fallback if no regions detected

    return s3im_val

def calculate_psnr(org_img: np.ndarray, pred_img: np.ndarray, data_range: float = None) -> float:
    """
    Peak Signal-to-Noise Ratio (PSNR).

    Args:
        org_img (np.ndarray): Reference image.
        pred_img (np.ndarray): Predicted image.
        data_range (float, optional): Data range of the input images. Defaults to the difference between max and min.

    Returns:
        float: PSNR value.
    """

    # # Debug: Check for NaN or empty images
    # print(f"PSNR input shapes - org_img: {org_img.shape}, pred_img: {pred_img.shape}")
    # print(f"PSNR input types - org_img: {org_img.dtype}, pred_img: {pred_img.dtype}")
    # print(f"PSNR input ranges - org_img: [{org_img.min()}, {org_img.max()}], pred_img: [{pred_img.min()}, {pred_img.max()}]")

    if data_range is None:
        data_range = org_img.max() - org_img.min()
    return peak_signal_noise_ratio(org_img, pred_img, data_range=data_range)


def calculate_ssim(org_img: np.ndarray, pred_img: np.ndarray, data_range: float = None) -> float:
    """
    Structural Similarity Index Measure (SSIM).

    Args:
        org_img (np.ndarray): Reference image.
        pred_img (np.ndarray): Predicted image.
        data_range (float, optional): Data range of the input images. Defaults to the difference between max and min.

    Returns:
        float: SSIM value.
    """
    # # Debug: Check for NaN or empty images
    # print(f"SSIM input shapes - org_img: {org_img.shape}, pred_img: {pred_img.shape}")
    # print(f"SSIM input types - org_img: {org_img.dtype}, pred_img: {pred_img.dtype}")
    # print(f"SSIM input ranges - org_img: [{org_img.min()}, {org_img.max()}], pred_img: [{pred_img.min()}, {pred_img.max()}]")

    if data_range is None:
        data_range = org_img.max() - org_img.min()
    return structural_similarity(org_img, pred_img, data_range=data_range)

def calculate_vifp(org_img: np.ndarray, pred_img: np.ndarray) -> float:
    """
    Visual Information Fidelity (VIFP) metric.

    Args:
        org_img (np.ndarray): Reference image.
        pred_img (np.ndarray): Predicted image.

    Returns:
        float: VIFP value.
    """
    # # Debug: Check for NaN or empty images
    # print(f"VIFP input shapes - org_img: {org_img.shape}, pred_img: {pred_img.shape}")
    # print(f"VIFP input types - org_img: {org_img.dtype}, pred_img: {pred_img.dtype}")
    # print(f"VIFP input ranges - org_img: [{org_img.min()}, {org_img.max()}], pred_img: [{pred_img.min()}, {pred_img.max()}]")

    return vifp(org_img, pred_img)




def calculate_uqi(org_img: np.ndarray, pred_img: np.ndarray) -> float:
    """
    Universal Quality Index (UQI).

    Args:
        org_img (np.ndarray): Reference image.
        pred_img (np.ndarray): Predicted image.

    Returns:
        float: UQI value.
    """
    # # Debug: Check for NaN or empty images
    # print(f"UQI input shapes - org_img: {org_img.shape}, pred_img: {pred_img.shape}")
    # print(f"UQI input types - org_img: {org_img.dtype}, pred_img: {pred_img.dtype}")
    # print(f"UQI input ranges - org_img: [{org_img.min()}, {org_img.max()}], pred_img: [{pred_img.min()}, {pred_img.max()}]")

    return uqi(org_img, pred_img)


def calculate_msssim(org_img: np.ndarray, pred_img: np.ndarray, ws) -> float:
    """
    Multi-Scale Structural Similarity (MS-SSIM) index.

    Args:
        org_img (np.ndarray): Reference image.
        pred_img (np.ndarray): Predicted image.

    Returns:
        float: MS-SSIM value.
    """
    # # Debug: Check for NaN or empty images
    # print(f"MSSIM input shapes - org_img: {org_img.shape}, pred_img: {pred_img.shape}")
    # print(f"MSSIM input types - org_img: {org_img.dtype}, pred_img: {pred_img.dtype}")
    # print(f"MSSIM input ranges - org_img: [{org_img.min()}, {org_img.max()}], pred_img: [{pred_img.min()}, {pred_img.max()}]")

    return msssim(org_img, pred_img, ws=ws)

# def gmsd(org_img: np.ndarray, pred_img: np.ndarray, rescale=True) -> float:
#     """
#     Gradient Magnitude Similarity Deviation (GMSD) metric.
    
#     Parameters:
#         org_img (np.ndarray): Reference image.
#         pred_img (np.ndarray): Image to compare.
#         rescale (bool): Whether to rescale images to a maximum value of 255.

#     Returns:
#         float: GMSD value.
#     """
#     if org_img.ndim == 3:
#         # If the images are multi-channel, calculate GMSD for each channel and average
#         gmsd_values = []
#         for i in range(org_img.shape[2]):
#             gmsd_value = GMSD(org_img[:, :, i], pred_img[:, :, i], rescale=rescale)
#             gmsd_values.append(gmsd_value)
#         return np.mean(gmsd_values)
#     else:
#         # Single-channel images
#         return GMSD(org_img, pred_img, rescale=rescale)

def _assert_image_shapes_equal(org_img: np.ndarray, pred_img: np.ndarray, metric: str):
    msg = (
        f"Cannot calculate {metric}. Input shapes not identical. y_true shape ="
        f" {str(org_img.shape)}, y_pred shape = {str(pred_img.shape)}"
    )
    assert org_img.shape == pred_img.shape, msg


def _similarity_measure(x: np.array, y: np.array, constant: float):
    """
    Calculate feature similarity measurement between two images.
    """
    numerator = 2 * np.multiply(x, y) + constant
    denominator = np.add(np.square(x), np.square(y)) + constant
    return np.divide(numerator, denominator)


def _gradient_magnitude(img: np.ndarray, img_depth: int):
    """
    Calculate gradient magnitude based on the Scharr operator.
    """
    scharrx = cv2.Scharr(img, img_depth, 1, 0)
    scharry = cv2.Scharr(img, img_depth, 0, 1)
    return np.sqrt(scharrx**2 + scharry**2)

def fsim(org_img: np.ndarray, pred_img: np.ndarray, T1: float = 0.85, T2: float = 160) -> float:
    """
    Feature-based similarity index, based on phase congruency (PC) and image gradient magnitude (GM).
    Works for both 2D and 3D images.

    :param org_img: Ground truth image (2D or 3D numpy array).
    :param pred_img: Predicted image (same shape as org_img).
    :param T1: Tuning parameter for phase congruency similarity.
    :param T2: Tuning parameter for gradient similarity.
    :return: FSIM score (float).
    """
    # # Debug: Check for NaN or empty images
    # print(f"FSIM input shapes - org_img: {org_img.shape}, pred_img: {pred_img.shape}")
    # print(f"FSIM input types - org_img: {org_img.dtype}, pred_img: {pred_img.dtype}")
    # print(f"FSIM input ranges - org_img: [{org_img.min()}, {org_img.max()}], pred_img: [{pred_img.min()}, {pred_img.max()}]")

    _assert_image_shapes_equal(org_img, pred_img, "FSIM")

    # If 2D image, directly compute FSIM
    if org_img.ndim == 2:
        pc1 = pc(org_img, nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978)
        pc2 = pc(pred_img, nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978)

        pc1_sum = sum(pc1[4])
        pc2_sum = sum(pc2[4])

        gm1 = _gradient_magnitude(org_img, cv2.CV_16U)
        gm2 = _gradient_magnitude(pred_img, cv2.CV_16U)

        S_pc = _similarity_measure(pc1_sum, pc2_sum, T1)
        S_g = _similarity_measure(gm1, gm2, T2)

        S_l = (S_pc**1) * (S_g**1)

        numerator = np.sum(S_l * np.maximum(pc1_sum, pc2_sum))
        denominator = np.sum(np.maximum(pc1_sum, pc2_sum))

        return numerator / denominator

    # If 3D image, iterate over slices
    fsim_list = []
    for i in range(org_img.shape[0]):  # Iterate over slices (assuming shape [slices, H, W])
        pc1 = pc(org_img[i], nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978)
        pc2 = pc(pred_img[i], nscale=4, minWaveLength=6, mult=2, sigmaOnf=0.5978)

        pc1_sum = sum(pc1[4])
        pc2_sum = sum(pc2[4])

        gm1 = _gradient_magnitude(org_img[i], cv2.CV_16U)
        gm2 = _gradient_magnitude(pred_img[i], cv2.CV_16U)

        S_pc = _similarity_measure(pc1_sum, pc2_sum, T1)
        S_g = _similarity_measure(gm1, gm2, T2)

        S_l = (S_pc**1) * (S_g**1)

        numerator = np.sum(S_l * np.maximum(pc1_sum, pc2_sum))
        denominator = np.sum(np.maximum(pc1_sum, pc2_sum))

        fsim_list.append(numerator / denominator)

    return np.mean(fsim_list)  # Return mean FSIM over slices