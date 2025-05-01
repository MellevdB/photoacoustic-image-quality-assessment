import torch
import piq
import torch.nn.functional as F

import os
import sys


def compute_piq_metrics(y_pred, y_true=None, image_ids=None, batch_size=256):

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