import numpy as np
from evaluation.metrics.calculate import calculate_metrics
from evaluation.preprocess.scale_clip import scale_and_clip
import h5py
import os

def process_msfd(data, dataset_info, full_config, results, metric_type):
    file_path = dataset_info["path"].get(file_key) if isinstance(dataset_info["path"], dict) else dataset_info["path"]
    if not os.path.isfile(file_path):
        print(f"[WARNING] File not found: {file_path}. Skipping...")
        return
    
    with h5py.File(file_path, 'r') as data:
        for wavelength, ground_truth_key in dataset_info["ground_truth"]["wavelengths"].items():
            print(f"Processing MSFD dataset with config={full_config}, wavelength={wavelength} and ground truth={ground_truth_key}")
            expected_key = f"{full_config}"
            if expected_key[-4:] == ground_truth_key[-4:]:
                y_pred = scale_and_clip(data[expected_key][:])
                y_true = scale_and_clip(data[ground_truth_key][:])
                metrics_mean, metrics_std, _ = calculate_metrics(y_pred, y_true, metric_type)
                results.append((full_config, ground_truth_key, wavelength, (metrics_mean, metrics_std)))