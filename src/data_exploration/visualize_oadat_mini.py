import os
import scipy.io as sio
import h5py
import numpy as np
from preprocessing_data.normalize import sigMatNormalize
from preprocessing_data.filterBandPass import sigMatFilter
from config.data_config import DATASETS
from .fr_metrics import (
    calculate_vifp,
    calculate_uqi,
    calculate_psnr,
    calculate_ssim,
    fsim,
)

# --- Utility Functions ---

def load_mat_file(file_path, key):
    """Load data from a MAT file."""
    try:
        data = sio.loadmat(file_path)
        return data[key]
    except NotImplementedError:
        with h5py.File(file_path, "r") as f:
            return f[key][()]

def process_data(data, key):
    """Normalize and filter data."""
    return sigMatNormalize(sigMatFilter(data[key][:]))

def calculate_metrics(y_pred, y_true):
    """Calculate image quality metrics."""
    data_range = y_true.max() - y_true.min()

    # PSNR and SSIM
    psnr = calculate_psnr(y_true, y_pred, data_range=data_range)
    ssim = calculate_ssim(y_true, y_pred, data_range=data_range)

    # Additional metrics
    vif = calculate_vifp(y_true, y_pred)
    fsim_score = fsim(y_true, y_pred)
    nqm = calculate_uqi(y_true, y_pred)

    metrics = {
        'PSNR': psnr,
        'SSIM': ssim,
        'VIF': vif,
        'FSIM': fsim_score,
        'NQM': nqm,
    }
    print("Calculated metrics:", metrics)
    return metrics

# --- Evaluation Functions ---

def evaluate_scd_swfd(data_path, configs, ground_truths):
    """Evaluate SCD and SWFD datasets."""
    results = []
    with h5py.File(data_path, "r") as data:
        for config, keys in configs.items():
            for full_config in keys:
                # Match the ground truth
                ground_truth_key = next((gt for gt, val in ground_truths.items() if gt in full_config), None)
                if not ground_truth_key:
                    print(f"[ERROR] No ground truth found for {full_config}")
                    continue

                # Check for the existence of data keys
                if full_config not in data or ground_truths[ground_truth_key] not in data:
                    print(f"[ERROR] Missing config or ground truth: {full_config}, {ground_truths[ground_truth_key]}")
                    continue

                # Process and evaluate
                y_pred = process_data(data, full_config)
                y_true = process_data(data, ground_truths[ground_truth_key])
                metrics = calculate_metrics(y_pred, y_true)
                results.append((full_config, ground_truths[ground_truth_key], metrics))
    return results

def evaluate_msfd(data_path, configs, ground_truths):
    """Evaluate MSFD datasets with wavelengths."""
    results = []
    with h5py.File(data_path, "r") as data:
        for config, keys in configs.items():
            for full_config in keys:
                # Find matching ground truth by wavelength
                matched_gt = None
                for wavelength, gt_key in ground_truths["wavelengths"].items():
                    if full_config.endswith(gt_key[-4:]):  # Match suffix
                        matched_gt = gt_key
                        break

                if not matched_gt:
                    print(f"[ERROR] No ground truth found for {full_config}")
                    continue

                # Check for the existence of data keys
                if full_config not in data or matched_gt not in data:
                    print(f"[ERROR] Missing config or ground truth: {full_config}, {matched_gt}")
                    continue

                # Process and evaluate
                y_pred = process_data(data, full_config)
                y_true = process_data(data, matched_gt)
                metrics = calculate_metrics(y_pred, y_true)
                results.append((full_config, matched_gt, wavelength, metrics))
    return results

def evaluate_mice_phantom(data_path, configs, ground_truth):
    """Evaluate mice, phantom, and v_phantom datasets."""
    results = []
    for config, keys in configs.items():
        for full_config in keys:
            config_file = os.path.join(data_path, f"{full_config}.mat")
            ground_truth_file = os.path.join(data_path, f"{ground_truth}.mat")

            if not os.path.isfile(config_file):
                print(f"[ERROR] Config file not found: {config_file}")
                continue
            if not os.path.isfile(ground_truth_file):
                print(f"[ERROR] Ground truth file not found: {ground_truth_file}")
                continue

            # Load and process data
            try:
                y_pred = load_mat_file(config_file, full_config)
                y_true = load_mat_file(ground_truth_file, ground_truth)
                y_pred = sigMatNormalize(sigMatFilter(y_pred))
                y_true = sigMatNormalize(sigMatFilter(y_true))
                metrics = calculate_metrics(y_pred, y_true)
                results.append((full_config, ground_truth, metrics))
            except Exception as e:
                print(f"[ERROR] Failed to evaluate {full_config} with {ground_truth}: {e}")
    return results

# --- Main Evaluation Dispatcher ---

def evaluate(dataset, config, full_config, file_key=None):
    dataset_info = DATASETS[dataset]
    data_path = dataset_info["path"]

    if dataset in ["SCD", "SWFD"]:
        file_path = data_path if isinstance(data_path, str) else data_path[file_key]
        return evaluate_scd_swfd(file_path, dataset_info["configs"], dataset_info["ground_truth"])
    elif dataset == "MSFD":
        return evaluate_msfd(data_path, dataset_info["configs"], dataset_info["ground_truth"])
    elif dataset in ["mice", "phantom", "v_phantom"]:
        return evaluate_mice_phantom(data_path, dataset_info["configs"], dataset_info["ground_truth"])
    else:
        print(f"[ERROR] Unknown dataset: {dataset}")
        return []