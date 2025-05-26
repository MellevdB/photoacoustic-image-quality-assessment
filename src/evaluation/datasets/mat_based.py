import os
import imageio
from evaluation.metrics.calculate import calculate_metrics
from evaluation.preprocess.scale_clip import scale_and_clip
import h5py
from scipy.io import loadmat
import numpy as np
from evaluation.preprocess.normalize import min_max_normalize_per_image

def load_mat_file(file_path, key):
    try:
        return loadmat(file_path)[key]
    except NotImplementedError:
        with h5py.File(file_path, 'r') as f:
            return f[key][()]


def process_mat_dataset(dataset, dataset_info, config, full_config, results, metric_type):
    path = dataset_info["path"]
    gt_path = os.path.join(path, f"{dataset}_full_recon.mat")
    pred_path = os.path.join(path, f"{dataset}_{config}_recon.mat")

    if not (os.path.isfile(gt_path) and os.path.isfile(pred_path)):
        print(f"[ERROR] Missing .mat files for config={config}. Skipping.")
        return

    # Load and scale
    y_true = min_max_normalize_per_image(load_mat_file(gt_path, dataset_info["ground_truth"]))
    y_pred = min_max_normalize_per_image(load_mat_file(pred_path, full_config))

    image_ids = [f"{full_config}_slice_{i}" for i in range(y_pred.shape[0])]

    # Calculate metrics using true image IDs
    metrics_mean, metrics_std, raw_metrics, image_ids = calculate_metrics(
        y_pred, y_true, metric_type, image_ids=image_ids, store_images=True
    )

    results.append((full_config, dataset_info["ground_truth"], (metrics_mean, metrics_std, raw_metrics, image_ids)))