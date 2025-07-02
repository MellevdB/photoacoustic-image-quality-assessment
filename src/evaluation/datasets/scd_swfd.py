import h5py
import os
import numpy as np
import imageio
from evaluation.metrics.calculate import calculate_metrics
from evaluation.preprocess.scale_clip import scale_and_clip
from evaluation.preprocess.normalize import min_max_normalize_per_image

def save_original_images(img_array, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(img_array):
        path = os.path.join(output_dir, f"{prefix}_slice_{i}.png")
        # Normalize to [0, 255] just for saving (no clipping!)
        scaled_img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        imageio.imwrite(path, (scaled_img * 255).astype(np.uint8))

def process_scd_swfd(dataset, dataset_info, full_config, file_key, results, metric_type):
    if isinstance(dataset_info["path"], dict):
        file_path = dataset_info["path"].get("recon")
        gt_path = dataset_info["path"].get("gt", file_path)
    else:
        file_path = dataset_info["path"]
        gt_path = file_path  # fallback to same file
    if not os.path.isfile(file_path):
        print(f"[WARNING] File not found: {file_path}. Skipping...")
        return

    print(f"[DEBUG] Opening file: {file_path}")
    with h5py.File(file_path, 'r') as data, h5py.File(gt_path, 'r') as gt_data:
        print(f"[DEBUG] Available keys in file: {list(data.keys())}")
        print(f"[DEBUG] Looking for config: {full_config}")
        print(f"[DEBUG] Ground truth mapping: {dataset_info['ground_truth']}")

        ground_truth_key = None
        for gt_prefix, gt_val in dataset_info["ground_truth"].items():
            if full_config.startswith(gt_prefix):
                ground_truth_key = gt_val
                break

        print(f"[DEBUG] Matched ground truth key: {ground_truth_key}")

        if not ground_truth_key or full_config not in data or ground_truth_key not in gt_data:
            print(f"[ERROR] Missing configuration {full_config} or ground truth {ground_truth_key}. Skipping...")
            return

        print(f"Processing dataset={dataset}, config={full_config} with ground truth={ground_truth_key}")

        # Normalize data before metric computation
        y_pred = min_max_normalize_per_image(data[full_config][:])
        y_true = min_max_normalize_per_image(gt_data[ground_truth_key][:])

        print(f"[DEBUG] y_pred shape: {y_pred.shape}, dtype: {y_pred.dtype}")
        print(f"[DEBUG] y_true shape: {y_true.shape}, dtype: {y_true.dtype}")
        print(f"[DEBUG] Starting metric calculation...")

        image_ids = [f"{full_config}_slice_{i}" for i in range(y_pred.shape[0])]

        try:
            metrics_mean, metrics_std, raw_metrics, image_ids = calculate_metrics(
                y_pred, y_true, metric_type, image_ids=image_ids, store_images=True
            )
        except RuntimeError as e:
            print(f"[ERROR] RuntimeError during metric computation: {e}")
            return

        results.append((full_config, ground_truth_key, (metrics_mean, metrics_std, raw_metrics, image_ids)))