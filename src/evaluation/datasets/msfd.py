import os
import h5py
import numpy as np
import imageio
from evaluation.metrics.calculate import calculate_metrics
from evaluation.preprocess.normalize import min_max_normalize_per_image


def process_msfd(dataset, dataset_info, full_config, results, metric_type):
    if isinstance(dataset_info["path"], dict):
        sparse_path = dataset_info["path"]["sparse"]
        gt_path = dataset_info["path"]["full"]
    else:
        sparse_path = gt_path = dataset_info["path"]
    if not os.path.isfile(sparse_path) or not os.path.isfile(gt_path):
        print(f"[WARNING] File(s) not found: {sparse_path} or {gt_path}. Skipping...")
        return

    with h5py.File(sparse_path, 'r') as sparse_data, h5py.File(gt_path, 'r') as gt_data:
        for wavelength, ground_truth_key in dataset_info["ground_truth"]["wavelengths"].items():
            expected_key = f"{full_config}"
            if expected_key.endswith(f"w{wavelength}"):
                print(f"Processing MSFD config={expected_key}, wavelength={wavelength}")

                y_pred = min_max_normalize_per_image(sparse_data[expected_key][:])
                y_true = min_max_normalize_per_image(gt_data[ground_truth_key][:])

                image_ids = [f"{expected_key}_w{wavelength}_slice_{i}" for i in range(y_pred.shape[0])]

                metrics_mean, metrics_std, raw_metrics, image_ids = calculate_metrics(
                    y_pred, y_true, metric_type, image_ids=image_ids, store_images=True
                )
                results.append((full_config, ground_truth_key, wavelength, (metrics_mean, metrics_std, raw_metrics, image_ids)))