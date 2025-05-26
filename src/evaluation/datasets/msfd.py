import os
import h5py
import numpy as np
import imageio
from evaluation.metrics.calculate import calculate_metrics
from evaluation.preprocess.normalize import min_max_normalize_per_image


def process_msfd(dataset, dataset_info, full_config, results, metric_type):
    file_path = dataset_info["path"].get(dataset) if isinstance(dataset_info["path"], dict) else dataset_info["path"]
    if not os.path.isfile(file_path):
        print(f"[WARNING] File not found: {file_path}. Skipping...")
        return

    with h5py.File(file_path, 'r') as data:
        for wavelength, ground_truth_key in dataset_info["ground_truth"]["wavelengths"].items():
            print(f"Processing MSFD dataset with config={full_config}, wavelength={wavelength} and ground truth={ground_truth_key}")
            expected_key = f"{full_config}"

            if expected_key[-4:] == ground_truth_key[-4:]:
                y_pred = min_max_normalize_per_image(data[expected_key][:])
                y_true = min_max_normalize_per_image(data[ground_truth_key][:])

                image_ids = [f"{expected_key}_w{wavelength}_slice_{i}" for i in range(y_pred.shape[0])]

                metrics_mean, metrics_std, raw_metrics, image_ids = calculate_metrics(
                    y_pred, y_true, metric_type, image_ids=image_ids, store_images=True
                )
                results.append((full_config, ground_truth_key, wavelength, (metrics_mean, metrics_std, raw_metrics, image_ids)))