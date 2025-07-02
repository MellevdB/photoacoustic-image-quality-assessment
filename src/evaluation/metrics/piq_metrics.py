import torch
import piq
import torch.nn.functional as F
import numpy as np

import os
import sys

from sewar.full_ref import uqi
from evaluation.metrics.fr import fsim, calculate_s3im


def compute_piq_metrics(y_pred, y_true=None, image_ids=None, batch_size=256):

    print("batch size:", batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if image_ids is None or len(image_ids) != len(y_pred):
        raise ValueError("`image_ids` must be provided and must match the number of images in y_pred.")

    # === Optional override for GPU-sensitive scenes ===
    if any(f"scene_600{i}" in image_ids[0] for i in range(5)):
        print("[INFO] Forcing CPU mode for high-memory scene by scene name.")
        device = torch.device("cpu")

    # Convert numpy arrays to torch tensors (initially on CPU for inspection)
    y_pred = torch.from_numpy(y_pred).unsqueeze(1)

    if y_true is not None:
        y_true = torch.from_numpy(y_true).unsqueeze(1)

    # === Check if image size is too big for GPU ===
    max_pixels = np.max([img.shape[-1] * img.shape[-2] for img in y_pred.numpy()])
    if max_pixels > 2_000_000:
        print(f"[INFO] Large image detected ({max_pixels} px), forcing CPU mode.")
        device = torch.device("cpu")
        if batch_size > 1:
            print("[INFO] Reducing batch size to 1 due to large image size on CPU.")
            batch_size = 1

    # Move tensors to correct device
    y_pred = y_pred.to(device)
    if y_true is not None:
        y_true = y_true.to(device)

    fr_metric_names = ['PSNR', 'SSIM', 'MSSSIM', 'IWSSIM', 'VIF', 'FSIM', 'GMSD', 'MSGMSD', 'HAARPSI', 'UQI', 'S3IM']
    nr_metric_names = ["TV", "BRISQUE", "CLIP-IQA"]
    all_metrics = {name: [] for name in fr_metric_names + nr_metric_names}

    clip_model = piq.CLIPIQA(data_range=1.0).to(device)

    print(f"[DEBUG] Total samples: {y_pred.shape[0]}, using batch size: {batch_size}")
    print(f"[DEBUG] Image shapes summary:")
    unique_shapes = set([img.shape for img in y_pred.cpu().numpy()])
    print(f"  Unique prediction image shapes: {unique_shapes}")
    if y_true is not None:
        unique_gt_shapes = set([img.shape for img in y_true.cpu().numpy()])
        print(f"  Unique GT image shapes: {unique_gt_shapes}")

    for i in range(0, y_pred.shape[0], batch_size):

        print(f"[DEBUG] Processing batch {i // batch_size} [{i}:{i + batch_size}]")
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

            tv_values = [
                piq.total_variation(img.unsqueeze(0), reduction='mean')
                for img in pred_batch
            ]
            all_metrics["TV"].append(torch.stack(tv_values).flatten())
            all_metrics["BRISQUE"].append(piq.brisque(pred_batch, data_range=1.0, reduction='none'))
            clip_scores = clip_model(pred_batch)
            all_metrics["CLIP-IQA"].append(clip_scores.flatten())

            for j in range(pred_batch.shape[0]):
                pred_img = pred_batch[j].squeeze().detach().cpu().numpy()
                print(f"[DEBUG] Image {j}: shape={pred_img.shape}, min={pred_img.min():.4f}, max={pred_img.max():.4f}, mean={pred_img.mean():.4f}")
                if pred_img.shape[0] < 32 or pred_img.shape[1] < 32:
                    print(f"[WARNING] Small image detected — some metrics may fail.")
                true_img = true_batch[j].squeeze().detach().cpu().numpy() if y_true is not None else None

                if true_img is not None:
                    try:
                        fsim_score = fsim(true_img, pred_img)
                    except Exception as e:
                        print(f"[FSIM ERROR] {e}")
                        fsim_score = float('nan')
                    all_metrics["FSIM"].append(fsim_score)

                    try:
                        uqi_score = uqi(true_img, pred_img)
                    except Exception as e:
                        print(f"[UQI ERROR] {e}")
                        uqi_score = float('nan')
                    all_metrics["UQI"].append(uqi_score)

                    try:
                        s3im_score = calculate_s3im(true_img, pred_img)
                    except Exception as e:
                        print(f"[S3IM ERROR] {e}")
                        s3im_score = float('nan')
                    all_metrics["S3IM"].append(s3im_score)

        except torch.cuda.OutOfMemoryError:
            print(f"[OOM] Skipping batch {i // batch_size} due to memory error.")
            for key in all_metrics:
                all_metrics[key].append(torch.full((pred_batch.shape[0],), float('nan'), device=device))

        torch.cuda.empty_cache()

    mean_metrics, std_metrics, raw_metrics = {}, {}, {}
    for name in fr_metric_names + nr_metric_names:
        if all_metrics[name]:
            if isinstance(all_metrics[name][0], torch.Tensor):
                values = torch.cat(all_metrics[name])
            else:
                values = torch.tensor(all_metrics[name], device=device)

            raw_metrics[name] = values.cpu().numpy()
            mean_metrics[name] = values.mean().item()
            std_metrics[name] = values.std().item()
            print(f"Raw metric '{name}': shape {values.shape}, sample: {values[:5]}")

            # Normalize selected metrics
            if name == "PSNR":
                raw_metrics["PSNR_norm"] = np.clip(raw_metrics[name] / 80.0, 0, 1)
            elif name in ["GMSD", "MSGMSD"]:
                raw_metrics[name + "_norm"] = np.clip(raw_metrics[name] / 0.35, 0, 1)
            elif name == "BRISQUE":
                brisque_vals = raw_metrics[name]
                brisque_min = brisque_vals.min()
                brisque_max = brisque_vals.max()
                norm = (brisque_vals - brisque_min) / (brisque_max - brisque_min + 1e-8)
                raw_metrics["BRISQUE_norm"] = np.clip(norm, 0, 1)
        else:
            raw_metrics[name] = None
            mean_metrics[name] = float('nan')
            std_metrics[name] = float('nan')

    if image_ids is None:
        raise ValueError("`image_ids` must be provided and must match the number of images in y_pred.")

    if "RECON_IMAGE" not in raw_metrics:
        raw_metrics["RECON_IMAGE"] = y_pred.squeeze(1).cpu().numpy()

    return mean_metrics, std_metrics, raw_metrics, image_ids