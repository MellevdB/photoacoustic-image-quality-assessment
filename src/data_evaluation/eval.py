import os
import scipy.io as sio
import h5py
import numpy as np
from preprocessing_data.normalize import sigMatNormalize
from preprocessing_data.filterBandPass import sigMatFilter
from config.data_config import DATASETS, DATA_DIR, RESULTS_DIR
from .fr_metrics import (
    fsim,
    calculate_vifp,
    calculate_uqi,
    calculate_psnr,
    calculate_ssim,
    calculate_s3im
)

from .nr_metrics import (
    calculate_brisque,
    calculate_niqe,
    calculate_niqe_k
)

def load_mat_file(file_path, key):
    """
    Load data from a MATLAB .mat file. Handles both v7.3 (h5py) and older versions (scipy.io).
    """
    try:
        data = sio.loadmat(file_path)
        return data[key]
    except NotImplementedError:
        with h5py.File(file_path, "r") as f:
            return f[key][()]

def calculate_metrics(y_pred, y_true, metric_type="all", fake_results=False):
    """
    Calculate image quality metrics based on `metric_type` while also computing 
    standard deviations across slices.

    :param y_pred: Predicted image (numpy array [slices, height, width]).
    :param y_true: Ground truth image (same shape as y_pred).
    :param metric_type: "fr" (full-reference), "nr" (no-reference), or "all".
    :param fake_results: If True, return random dummy values.
    :return: Tuple containing mean and standard deviation dictionaries.
    """
    if fake_results:
        print("⚠️ Using FAKE metric values for quick testing!")
        metrics_mean = {key: np.random.uniform(0.1, 1.0) for key in ['PSNR', 'SSIM', 'VIF', 'FSIM', 'UQI', 'S3IM', 'BRISQUE']}
        metrics_std = {key: np.random.uniform(0.01, 0.1) for key in metrics_mean}
        return metrics_mean, metrics_std

    num_slices = y_pred.shape[0]

    if metric_type in ["fr", "all"]:
        data_range = y_true.max() - y_true.min()
        fr_metrics = {
            'PSNR': calculate_psnr(y_true, y_pred, data_range=data_range),
            'SSIM': calculate_ssim(y_true, y_pred, data_range=data_range),
            'VIF': calculate_vifp(y_true, y_pred),
            'FSIM': fsim(y_true, y_pred),
            'UQI': calculate_uqi(y_true, y_pred),
            'S3IM': calculate_s3im(y_true, y_pred)
        }
        fr_std = {key: np.std([calculate_psnr(y_true[i], y_pred[i], data_range) if key == 'PSNR'
                               else calculate_ssim(y_true[i], y_pred[i], data_range) if key == 'SSIM'
                               else calculate_vifp(y_true[i], y_pred[i]) if key == 'VIF'
                               else fsim(y_true[i], y_pred[i]) if key == 'FSIM'
                               else calculate_uqi(y_true[i], y_pred[i]) if key == 'UQI'
                               else calculate_s3im(y_true[i], y_pred[i])
                               for i in range(num_slices)])
                  for key in fr_metrics}
    else:
        fr_metrics, fr_std = {}, {}

    if metric_type in ["nr", "all"]:
        per_slice_metrics = {'BRISQUE': [], 'NIQE': [], 'NIQE-K': []}
        for i in range(num_slices):
            slice_img = y_pred[i]
            per_slice_metrics['BRISQUE'].append(calculate_brisque(slice_img))
            per_slice_metrics['NIQE'].append(calculate_niqe(slice_img))
            per_slice_metrics['NIQE-K'].append(calculate_niqe_k(slice_img))
        nr_metrics = {k: np.mean(v) for k, v in per_slice_metrics.items()}
        nr_std = {k: np.std(v) for k, v in per_slice_metrics.items()}
    else:
        nr_metrics, nr_std = {}, {}

    metrics_mean = {**fr_metrics, **nr_metrics}
    metrics_std = {**fr_std, **nr_std}

    return metrics_mean, metrics_std

def evaluate(dataset, config, full_config, file_key=None, metric_type="all", fake_results=False):
    """
    Evaluate a specific dataset and configuration for PSNR, SSIM, etc.

    :param metric_type: "fr", "nr", or "all".
    :param fake_results: If True, return fake metrics.
    :return: List of tuples.
    """
    results = []
    dataset_info = DATASETS[dataset]

    print(f"Processing dataset={dataset}, config={full_config}")

    if dataset in ["SCD", "SWFD", "MSFD"]:
        _process_hdf5_dataset(dataset, dataset_info, full_config, file_key, results, metric_type, fake_results)
    elif dataset in ["mice", "phantom", "v_phantom"]:
        _process_mat_dataset(dataset, dataset_info, config, full_config, results, metric_type, fake_results)

    return results

def _process_hdf5_dataset(dataset, dataset_info, full_config, file_key, results, metric_type, fake_results):
    """Process HDF5-based datasets (SCD, SWFD, MSFD)."""
    data_path = dataset_info["path"].get(file_key) if isinstance(dataset_info["path"], dict) else dataset_info["path"]
    if not os.path.isfile(data_path):
        print(f"[WARNING] File not found: {data_path}. Skipping...")
        return

    with h5py.File(data_path, "r") as data:
        if dataset == "MSFD":
            _evaluate_msfd(data, dataset_info, full_config, results, metric_type, fake_results)
        else:
            _evaluate_scd_swfd(data, dataset, dataset_info, full_config, file_key, results, metric_type, fake_results)

def _evaluate_msfd(data, dataset_info, full_config, results, metric_type, fake_results):
    """Evaluate MSFD dataset by iterating over each wavelength."""
    for wavelength, ground_truth_key in dataset_info["ground_truth"]["wavelengths"].items():
        expected_key = f"{full_config}"
        if expected_key[-4:] == ground_truth_key[-4:]:
            y_pred = np.random.rand(128, 128, 128) if fake_results else sigMatNormalize(sigMatFilter(data[expected_key][:]))
            y_true = np.random.rand(128, 128, 128) if fake_results else sigMatNormalize(sigMatFilter(data[ground_truth_key][:]))
            metrics = calculate_metrics(y_pred, y_true, metric_type, fake_results)
            results.append((full_config, ground_truth_key, wavelength, metrics))
        else:
            print(f"Skipping mismatched keys: {expected_key} vs {ground_truth_key}")

def _evaluate_scd_swfd(data, dataset, dataset_info, full_config, file_key, results, metric_type, fake_results):
    """Evaluate SCD and SWFD datasets."""
    ground_truth_key = dataset_info["ground_truth"].get(file_key) if file_key else next(
        (dataset_info["ground_truth"][key] for key in dataset_info["ground_truth"] if key in full_config), None)

    if not ground_truth_key or full_config not in data or ground_truth_key not in data:
        print(f"[ERROR] Missing configuration {full_config} or ground truth {ground_truth_key}. Skipping...")
        return

    y_pred = np.random.rand(128, 128, 128) if fake_results else sigMatNormalize(sigMatFilter(data[full_config][:]))
    y_true = np.random.rand(128, 128, 128) if fake_results else sigMatNormalize(sigMatFilter(data[ground_truth_key][:]))
    metrics = calculate_metrics(y_pred, y_true, metric_type, fake_results)
    results.append((full_config, ground_truth_key, metrics))

def _process_mat_dataset(dataset, dataset_info, config, full_config, results, metric_type, fake_results):
    """Process MAT-file-based datasets (mice, phantom, v_phantom)."""
    path = dataset_info["path"]
    gt_file, config_file = os.path.join(path, f"{dataset}_full_recon.mat"), os.path.join(path, f"{dataset}_{config}_recon.mat")

    if not all(map(os.path.isfile, [gt_file, config_file])):
        print(f"[WARNING] Missing MAT files for {dataset}. Skipping...")
        return

    y_pred = np.random.rand(128, 128, 128) if fake_results else sigMatNormalize(sigMatFilter(load_mat_file(config_file, full_config)))
    y_true = np.random.rand(128, 128, 128) if fake_results else sigMatNormalize(sigMatFilter(load_mat_file(gt_file, dataset_info["ground_truth"])))
    metrics = calculate_metrics(y_pred, y_true, metric_type, fake_results)
    results.append((full_config, dataset_info["ground_truth"], metrics))