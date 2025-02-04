import os
import scipy.io as sio
import h5py
import numpy as np
from preprocessing_data.normalize import sigMatNormalize
from preprocessing_data.filterBandPass import sigMatFilter
from config.data_config import DATASETS, DATA_DIR, RESULTS_DIR
from .metrics import (
    fsim,
    # gmsd,
    calculate_vifp,
    calculate_uqi,
    # calculate_msssim,
    calculate_psnr,
    calculate_ssim,
    calculate_s3im
)

def load_mat_file(file_path, key):
    """
    Load data from a MATLAB .mat file. This handles both v7.3 (h5py) and older versions (scipy.io).
    :param file_path: Path to the .mat file.
    :param key: Key inside the .mat file to load.
    :return: Numpy array of the data for the provided key.
    """
    try:
        data = sio.loadmat(file_path)
        return data[key]
    except NotImplementedError:
        with h5py.File(file_path, "r") as f:
            return f[key][()]

def calculate_metrics(y_pred, y_true):
    """
    Calculate image quality metrics for the full volume while also computing 
    the standard deviation across slices.

    :param y_pred: Predicted image (numpy array of shape [slices, height, width]).
    :param y_true: Ground truth image (same shape as y_pred).
    :return: Tuple containing:
             - Dictionary of mean metrics for the whole volume
             - Dictionary of standard deviations across slices
    """
    num_slices = y_pred.shape[0]

    # Ensure valid data range
    data_range = y_true.max() - y_true.min()

    # === Compute Volume-wise Metrics (Entire 3D Image at Once) ===
    volume_metrics = {
        'PSNR': calculate_psnr(y_true, y_pred, data_range=data_range),
        'SSIM': calculate_ssim(y_true, y_pred, data_range=data_range),
        'VIF': calculate_vifp(y_true, y_pred),
        'FSIM': fsim(y_true, y_pred),
        'NQM': calculate_uqi(y_true, y_pred),
        'S3IM': calculate_s3im(y_true, y_pred)
    }

    # === Compute Per-Slice Metrics for Standard Deviation ===
    all_metrics = {key: [] for key in volume_metrics}  # Store slice-wise values

    for i in range(num_slices):
        all_metrics['PSNR'].append(calculate_psnr(y_true[i], y_pred[i], data_range=data_range))
        all_metrics['SSIM'].append(calculate_ssim(y_true[i], y_pred[i], data_range=data_range))
        all_metrics['VIF'].append(calculate_vifp(y_true[i], y_pred[i]))
        all_metrics['FSIM'].append(fsim(y_true[i], y_pred[i]))
        all_metrics['NQM'].append(calculate_uqi(y_true[i], y_pred[i]))
        all_metrics['S3IM'].append(calculate_s3im(y_true[i], y_pred[i]))

    # Compute standard deviation across slices
    std_metrics = {key: np.std(all_metrics[key]) for key in all_metrics}

    print("Volume Metrics:", volume_metrics)
    print("Per-Slice STD:", std_metrics)

    return volume_metrics, std_metrics

def evaluate(dataset, config, full_config, file_key=None, save_results=True):
    """
    Evaluate a specific dataset and configuration for PSNR, SSIM, etc.

    :param dataset: Dataset name (e.g., SCD, SWFD, MSFD, mice, phantom, v_phantom).
    :param config: Top-level configuration (e.g. 'vc', 'lv128_BP').
    :param full_config: Full config key (includes resolution, etc.).
    :param file_key: Optional key for datasets like SWFD (e.g., "multisegment" or "semicircle").
    :param save_results: Whether to save the results to a file (not used here, but kept for compatibility).
    :return: List of tuples:
        - For MSFD: (full_config, ground_truth_key, wavelength, metrics)
        - For others: (full_config, ground_truth_key, metrics)
    """
    results = []
    dataset_info = DATASETS[dataset]

    if dataset == "SWFD":
        print(f"Processing dataset={dataset}, config={full_config}, file_key={file_key}")
    else:
        print(f"Processing dataset={dataset}, config={full_config}")

    # Handle HDF5-based datasets
    if dataset in ["SCD", "SWFD", "MSFD"]:
        _process_hdf5_dataset(dataset, dataset_info, full_config, file_key, results)

    # Handle MAT file-based datasets
    elif dataset in ["mice", "phantom", "v_phantom"]:
        _process_mat_dataset(dataset, dataset_info, config, full_config, results)

    return results

def _process_hdf5_dataset(dataset, dataset_info, full_config, file_key, results):
    """
    Internal helper for evaluating HDF5-based datasets (SCD, SWFD, MSFD).
    """
    # For datasets with multiple paths (like SWFD), get the correct path
    if isinstance(dataset_info["path"], dict):
        data_path = dataset_info["path"].get(file_key)
        if not data_path:
            print(f"[WARNING] File key '{file_key}' not found for dataset '{dataset}'. Skipping...")
            return
    else:
        # Single path
        data_path = dataset_info["path"]

    if not os.path.isfile(data_path):
        print(f"[WARNING] File not found: {data_path}. Skipping...")
        return

    with h5py.File(data_path, "r") as data:
        if dataset == "MSFD":
            _evaluate_msfd(data, dataset_info, full_config, results)
        else:
            _evaluate_scd_swfd(data, dataset, dataset_info, full_config, file_key, results)

def _evaluate_msfd(data, dataset_info, full_config, results):
    """
    Evaluate the MSFD dataset by iterating over each wavelength defined in the config.
    Uses 70% of the slices for training (metric calculation).
    """
    for wavelength, ground_truth_key in dataset_info["ground_truth"]["wavelengths"].items():
        expected_key = f"{full_config}"
        
        if expected_key[-4:] == ground_truth_key[-4:]:
            y_pred = sigMatNormalize(sigMatFilter(data[expected_key][:]))
            y_true = sigMatNormalize(sigMatFilter(data[ground_truth_key][:]))
            
            # Split the data into training (70%), validation (15%), and test (15%)
            num_slices = y_true.shape[-1]
            train_size = int(0.7 * num_slices)
            
            # Randomly select training slices
            train_indices = np.random.choice(num_slices, train_size, replace=False)
            y_pred_train = y_pred[..., train_indices]
            y_true_train = y_true[..., train_indices]
            
            # Calculate volume-wise metrics on training data
            volume_metrics = calculate_metrics(y_pred_train, y_true_train)
            
            # Initialize dictionaries to store per-slice metrics for std calculation
            slice_metrics = {
                'PSNR': [], 'SSIM': [], 'VIF': [], 
                'FSIM': [], 'NQM': [], 'MSSIM': [],
                'GMSD': [], 'HDRVDP': []
            }
            
            # Calculate per-slice metrics only for std (on training data)
            for slice_idx in range(y_true_train.shape[-1]):
                slice_result = calculate_metrics(y_pred_train[..., slice_idx], y_true_train[..., slice_idx])
                for metric in slice_metrics.keys():
                    if f'{metric}_mean' in slice_result:
                        slice_metrics[metric].append(slice_result[f'{metric}_mean'])
            
            # Combine volume metrics with slice std
            final_metrics = {}
            for metric in slice_metrics.keys():
                if slice_metrics[metric]:  # If we have values for this metric
                    values = np.array(slice_metrics[metric])
                    final_metrics[f'{metric}_mean'] = volume_metrics[f'{metric}_mean']  # Use volume-wise mean
                    final_metrics[f'{metric}_std'] = float(np.std(values))  # Use per-slice std
            
            results.append((full_config, ground_truth_key, wavelength, final_metrics))
        else:
            print(f"Key not corresponding to correct ground truth: {expected_key} is not the same wavelength as {ground_truth_key}")

def _evaluate_scd_swfd(data, dataset, dataset_info, full_config, file_key, results):
    """
    Evaluate the SCD or SWFD dataset by finding the correct ground truth key
    and computing the metrics against the predicted data.
    """
    ground_truth_key = None
    if file_key:
        # For SWFD specifically
        ground_truth_key = dataset_info["ground_truth"].get(file_key)
    else:
        # For SCD, match ground_truth mapping by checking if the config name is in a key
        for key in dataset_info["ground_truth"]:
            if key in full_config:
                ground_truth_key = dataset_info["ground_truth"][key]
                break

    if not ground_truth_key:
        print(f"[ERROR] Ground truth key not found for config '{full_config}' and file_key '{file_key}'. Skipping...")
        return

    if full_config in data and ground_truth_key in data:
        y_pred = sigMatNormalize(sigMatFilter(data[full_config][:]))
        y_true = sigMatNormalize(sigMatFilter(data[ground_truth_key][:]))
        metrics = calculate_metrics(y_pred, y_true)
        results.append((full_config, ground_truth_key, metrics))
    else:
        print(f"[ERROR] Configuration '{full_config}' or ground truth '{ground_truth_key}' not found in data. Skipping...")

def _process_mat_dataset(dataset, dataset_info, config, full_config, results):
    """
    Internal helper for evaluating MAT-file-based datasets (mice, phantom, v_phantom).
    """
    path = dataset_info["path"]
    gt_file = os.path.join(path, f"{dataset}_full_recon.mat")
    config_file = os.path.join(path, f"{dataset}_{config}_recon.mat")

    if not os.path.isfile(gt_file):
        print(f"[WARNING] Ground truth file not found: {gt_file}. Skipping...")
        return

    if not os.path.isfile(config_file):
        print(f"[WARNING] Configuration file not found: {config_file}. Skipping...")
        return

    # Load ground truth and configuration data
    gt_data = load_mat_file(gt_file, dataset_info["ground_truth"])
    config_data = load_mat_file(config_file, full_config)

    print(f"Processing config={full_config}")
    print(f"Ground truth key: {dataset_info['ground_truth']}")
    y_pred = sigMatNormalize(sigMatFilter(config_data))
    y_true = sigMatNormalize(sigMatFilter(gt_data))

    metrics = calculate_metrics(y_pred, y_true)
    results.append((full_config, dataset_info["ground_truth"], metrics))