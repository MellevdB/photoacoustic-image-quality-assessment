import torch
import piq
import torch.nn.functional as F

import os
import sys

# Compute the absolute path to third_party/pytorch_fsim from the current file.
module_parent_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../third_party'))
if module_parent_path not in sys.path:
    sys.path.append(module_parent_path)

from pytorch_fsim.fsim import FSIMc

def compute_piq_metrics(y_pred, y_true=None, image_ids=None, batch_size=256):
    import torch
    import piq
    from pytorch_fsim.fsim import FSIMc

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def normalize_for_piq(tensor):
        tensor = tensor.float()
        min_val = tensor.min()
        max_val = tensor.max()
        if min_val < 0.0 or max_val > 1.0:
            print(f"Normalizing tensor with min={min_val.item():.3f}, max={max_val.item():.3f} to [0, 1]")
            return (tensor - min_val) / (max_val - min_val + 1e-8)
        print("Tensor already in [0, 1] range — no normalization needed.")
        return tensor

    if image_ids is None or len(image_ids) != len(y_pred):
        raise ValueError("`image_ids` must be provided and must match the number of images in y_pred.")

    # Convert numpy arrays to torch tensors
    y_pred = torch.from_numpy(y_pred).unsqueeze(1).to(device)
    y_pred = normalize_for_piq(y_pred)

    if y_true is not None:
        y_true = torch.from_numpy(y_true).unsqueeze(1).to(device)
        y_true = normalize_for_piq(y_true)

    fr_metric_names = ['PSNR', 'SSIM', 'MSSSIM', 'IWSSIM', 'VIF', 'FSIM', 'GMSD', 'MSGMSD', 'HAARPSI','UQI', 'S3IM']
    nr_metric_names = ["TV", "BRISQUE", "CLIP-IQA"]
    all_metrics = {name: [] for name in fr_metric_names + nr_metric_names}
    
    clip_model = piq.CLIPIQA(data_range=1.0).to(device)
    fsim_loss = FSIMc()  # optional

    for i in range(0, y_pred.shape[0], batch_size):
        pred_batch = y_pred[i:i+batch_size]
        if y_true is not None:
            true_batch = y_true[i:i+batch_size]

        try:
            if y_true is not None:
                all_metrics["PSNR"].append(piq.psnr(pred_batch, true_batch, data_range=1.0, reduction='none'))
                all_metrics["SSIM"].append(piq.ssim(pred_batch, true_batch, data_range=1.0, reduction='none'))
                all_metrics["MSSSIM"].append(piq.multi_scale_ssim(pred_batch, true_batch, data_range=1.0, reduction='none'))
                all_metrics["IWSSIM"].append(piq.information_weighted_ssim(pred_batch, true_batch, data_range=1.0, reduction='none'))
                all_metrics["VIF"].append(piq.vif_p(pred_batch, true_batch, data_range=1.0, reduction='none'))
                all_metrics["GMSD"].append(piq.gmsd(pred_batch, true_batch, data_range=1.0, reduction='none'))
                all_metrics["MSGMSD"].append(piq.multi_scale_gmsd(pred_batch, true_batch, data_range=1.0, reduction='none'))
                all_metrics["HAARPSI"].append(piq.haarpsi(pred_batch, true_batch, data_range=1.0, reduction='none'))
                # all_metrics["UQI"].append(calculate_uqi_torch(...))
                # all_metrics["S3IM"].append(s3im_torch(...))

            tv_values = [piq.total_variation(img.unsqueeze(0)) for img in pred_batch]
            all_metrics["TV"].append(torch.stack(tv_values).flatten())
            all_metrics["BRISQUE"].append(piq.brisque(pred_batch, data_range=1.0, reduction='none'))
            clip_scores = clip_model(pred_batch)
            all_metrics["CLIP-IQA"].append(clip_scores.flatten())

        except torch.cuda.OutOfMemoryError:
            print(f"[OOM] Skipping batch {i // batch_size} due to memory error.")
            for key in all_metrics:
                all_metrics[key].append(torch.full((pred_batch.shape[0],), float('nan'), device=device))

        torch.cuda.empty_cache()

    mean_metrics, std_metrics, raw_metrics = {}, {}, {}
    for key, value in all_metrics.items():
        print(f"Metric: {key}, Total values: {sum(v.shape[0] for v in value)}")

    for name in fr_metric_names + nr_metric_names:
        if all_metrics[name]:
            values = torch.cat(all_metrics[name])
            raw_metrics[name] = values.cpu().numpy()
            mean_metrics[name] = values.mean().item()
            std_metrics[name] = values.std().item()
            print(f"Raw metric '{name}': shape {values.shape}, sample: {values[:5]}")
        else:
            raw_metrics[name] = None
            mean_metrics[name] = float('nan')
            std_metrics[name] = float('nan')


    # If no image_ids provided, use default slice-based indexing
    if image_ids is None:
        raise ValueError("`image_ids` must be provided and must match the number of images in y_pred.")

    # Add reconstructed images to raw_metrics so they can be saved later
    if "RECON_IMAGE" not in raw_metrics:
        raw_metrics["RECON_IMAGE"] = y_pred.squeeze(1).cpu().numpy()  # shape: (N, H, W)
        
    return mean_metrics, std_metrics, raw_metrics, image_ids

def calculate_uqi_torch(org_img: torch.Tensor, pred_img: torch.Tensor, kernel_size: int = 8) -> torch.Tensor:
    """
    Compute the Universal Quality Index (UQI) on a batch of images.
    
    Args:
        org_img (torch.Tensor): Ground truth image tensor of shape (B, 1, H, W).
        pred_img (torch.Tensor): Predicted image tensor of the same shape.
        kernel_size (int): Window size used for local statistics (default=8).
    
    Returns:
        torch.Tensor: Mean UQI value over the batch.
    """
    # Compute local means using average pooling.
    mu_x = F.avg_pool2d(org_img, kernel_size, stride=1, padding=kernel_size//2)
    mu_y = F.avg_pool2d(pred_img, kernel_size, stride=1, padding=kernel_size//2)
    
    # Compute local variances and covariance.
    sigma_x_sq = F.avg_pool2d(org_img ** 2, kernel_size, stride=1, padding=kernel_size//2) - mu_x ** 2
    sigma_y_sq = F.avg_pool2d(pred_img ** 2, kernel_size, stride=1, padding=kernel_size//2) - mu_y ** 2
    sigma_xy = F.avg_pool2d(org_img * pred_img, kernel_size, stride=1, padding=kernel_size//2) - mu_x * mu_y
    
    numerator = 4 * mu_x * mu_y * sigma_xy
    denominator = (mu_x ** 2 + mu_y ** 2) * (sigma_x_sq + sigma_y_sq) + 1e-8
    
    uqi_map = numerator / denominator
    return uqi_map.mean()

#def fsim_torch(org_img: torch.Tensor, pred_img: torch.Tensor, T2: float = 160) -> torch.Tensor:
#     """
#     Approximated FSIM calculation using gradient magnitude similarity.
#     Note: This version does not include phase congruency; it relies solely
#     on gradient similarity. In practice, FSIM requires also computing
#     phase congruency maps.
    
#     Args:
#         org_img (torch.Tensor): Ground truth image tensor (B, 1, H, W)
#         pred_img (torch.Tensor): Predicted image tensor (B, 1, H, W)
#         T2 (float): Tuning constant for gradient similarity (default=160)
    
#     Returns:
#         torch.Tensor: Mean FSIM score over the batch.
#     """
#     device = org_img.device
#     # Define Scharr kernels for x and y directions.
#     scharr_kernel_x = torch.tensor([[3, 0, -3],
#                                     [10, 0, -10],
#                                     [3, 0, -3]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
#     scharr_kernel_y = torch.tensor([[3, 10, 3],
#                                     [0, 0, 0],
#                                     [-3, -10, -3]], dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
    
#     # Compute gradient maps with padding.
#     gm1_x = F.conv2d(org_img, scharr_kernel_x, padding=1)
#     gm1_y = F.conv2d(org_img, scharr_kernel_y, padding=1)
#     gm1 = torch.sqrt(gm1_x ** 2 + gm1_y ** 2 + 1e-8)
    
#     gm2_x = F.conv2d(pred_img, scharr_kernel_x, padding=1)
#     gm2_y = F.conv2d(pred_img, scharr_kernel_y, padding=1)
#     gm2 = torch.sqrt(gm2_x ** 2 + gm2_y ** 2 + 1e-8)
    
#     # Compute gradient similarity (similar to the similarity measure defined below).
#     S_g = (2 * gm1 * gm2 + T2) / (gm1 ** 2 + gm2 ** 2 + T2)
    
#     # Return the mean similarity as a proxy for FSIM.
#     return S_g.mean()

def s3im_torch(org_img: torch.Tensor, pred_img: torch.Tensor, sensitivity: float = 0.5, window_size: int = 11) -> torch.Tensor:
    """
    Approximated S3IM computation in PyTorch using a vectorized approach.
    Uses nn.Unfold to extract sliding-window patches and computes per-patch SSIM,
    then averages based on a mask computed from local averages.
    
    Args:
        org_img (torch.Tensor): Ground truth images (B, 1, H, W)
        pred_img (torch.Tensor): Predicted images (B, 1, H, W)
        sensitivity (float): Factor to adjust the threshold (default=0.5)
        window_size (int): Size of the sliding window (default=11)
    
    Returns:
        torch.Tensor: Mean S3IM value over the batch.
    """
    # Ensure inputs have shape (B, 1, H, W)
    if org_img.dim() == 3:
        org_img = org_img.unsqueeze(1)
    if pred_img.dim() == 3:
        pred_img = pred_img.unsqueeze(1)
    
    B, C, H, W = org_img.shape
    padding = window_size // 2
    
    # Extract sliding-window patches using nn.Unfold
    unfold = torch.nn.Unfold(kernel_size=window_size, padding=padding, stride=1)
    patches_org = unfold(org_img)   # Shape: (B, window_size*window_size, L) where L = H*W
    patches_pred = unfold(pred_img) # Same shape
    
    L = patches_org.shape[-1]  # Number of patches per image (should equal H*W)
    
    # Reshape patches to (B*L, 1, window_size, window_size)
    patches_org = patches_org.transpose(1, 2).reshape(-1, 1, window_size, window_size)
    patches_pred = patches_pred.transpose(1, 2).reshape(-1, 1, window_size, window_size)
    
    # Compute SSIM on each patch; note: reduction can behave differently depending on input.
    ssim_patch = piq.ssim(patches_org, patches_pred, data_range=1.0, reduction='none')
    
    # Check the dimensionality of ssim_patch:
    if ssim_patch.dim() >= 3:
        # If the result is a map per patch, average over spatial dimensions.
        ssim_vals = ssim_patch.mean(dim=(-2, -1))
    else:
        # Assume ssim_patch already returns one scalar per patch.
        ssim_vals = ssim_patch
    
    # Ensure ssim_vals is of shape (B*L,)
    ssim_vals = ssim_vals.reshape(B, H, W).unsqueeze(1)
    
    # Compute local averages using average pooling (as a surrogate for adaptive thresholding)
    local_mean_org = F.avg_pool2d(org_img, window_size, stride=1, padding=padding)
    local_mean_pred = F.avg_pool2d(pred_img, window_size, stride=1, padding=padding)
    
    # Create a binary mask: mark pixels where either image is above (local average * sensitivity)
    mask = ((org_img > local_mean_org * sensitivity) | (pred_img > local_mean_pred * sensitivity)).float()
    
    # If needed, adjust ssim_map dimensions (they should match org_img)
    if ssim_vals.shape[-2:] != org_img.shape[-2:]:
        ssim_vals = F.interpolate(ssim_vals, size=org_img.shape[-2:], mode='bilinear', align_corners=False)
    
    # Apply the mask and compute the weighted average
    s3im_map = ssim_vals * mask
    s3im_val = s3im_map.sum() / (mask.sum() + 1e-8)
    
    return s3im_val