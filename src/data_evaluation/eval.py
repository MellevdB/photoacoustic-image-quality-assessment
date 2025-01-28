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
)


def load_mat_file(file_path, key):
    try:
        data = sio.loadmat(file_path)
        return data[key]
    except NotImplementedError:
        with h5py.File(file_path, "r") as f:
            return f[key][()]


def calculate_metrics(y_pred, y_true):
    """
    Calculate image quality metrics for the given predicted and ground truth images.
    
    :param y_pred: Predicted image.
    :param y_true: Ground truth image.
    :return: Dictionary containing all calculated metrics.
    """
    data_range = y_true.max() - y_true.min()

    # PSNR and SSIM
    psnr = calculate_psnr(y_true, y_pred, data_range=data_range)
    ssim = calculate_ssim(y_true, y_pred, data_range=data_range)

    # Additional metrics
    vif = calculate_vifp(y_true, y_pred)  # Visual Information Fidelity
    fsim_score = fsim(y_true, y_pred)  # Feature Similarity Index
    nqm = calculate_uqi(y_true, y_pred)  # Universal Quality Index (similar to NQM)


    # GMSD and HDRVDP placeholders (not implemented)
    gmsd_score = None  # Placeholder for GMSD
    hdrvdp_score = None  # Placeholder for HDRVDP
    iwssim = None  # Placeholder for IW-SSIM

    metrics = {
        'PSNR': psnr,
        'SSIM': ssim,
        'VIF': vif,
        'FSIM': fsim_score,
        'NQM': nqm,
        'MSSIM': iwssim,
        'GMSD': gmsd_score,  
        'HDRVDP': hdrvdp_score
    }
    print("Calculated metrics:", metrics)
    return metrics


def evaluate(dataset, config, full_config, file_key=None, save_results=True):
    """
    Evaluate a specific dataset and configuration for PSNR and SSIM.

    :param dataset: Dataset name (e.g., SCD, SWFD, MSFD, mice, phantom, v_phantom).
    :param full_config: Full configuration key (e.g., vc,lv128_BP).
    :param file_key: Optional key for datasets like SWFD (e.g., "multisegment" or "semicircle").
    :param save_results: Whether to save the results to a file.
    :return: List of tuples containing (configuration, psnr, ssim) or (configuration, wavelength, psnr, ssim) for MSFD.
    """
    results = []
    dataset_info = DATASETS[dataset]

    if dataset == "SWFD":
        print(f"Processing dataset={dataset}, config={full_config}, file_key={file_key}")
    else:
        print(f"Processing dataset={dataset}, config={full_config}")

    # Handle HDF5 datasets (e.g., SCD, SWFD, MSFD)
    if dataset in ["SCD", "SWFD", "MSFD"]:
        if isinstance(dataset_info["path"], dict):
            # For datasets with multiple file keys
            data_path = dataset_info["path"].get(file_key)
            if not data_path:
                print(f"[WARNING] File key '{file_key}' not found for dataset '{dataset}'. Skipping...")
                return results
        else:
            # For datasets with a single path
            data_path = dataset_info["path"]

        if not os.path.isfile(data_path):
            print(f"[WARNING] File not found: {data_path}. Skipping...")
            return results

        with h5py.File(data_path, "r") as data:
            if dataset == "MSFD":
                # MSFD: Iterate over wavelengths
                # print("Keys in data:", list(data.keys()))
                for wavelength, ground_truth_key in dataset_info["ground_truth"]["wavelengths"].items():
                    # Construct the correct key for the current configuration and wavelength
                    expected_key = f"{full_config}"
                    print(f"Checking key: {expected_key}")
                    print(f"Ground truth key: {ground_truth_key}")
                    
                    # print("last 4 characters of expected_key", expected_key[-4:])
                    # print("last 4 characters of ground_truth_key", ground_truth_key[-4:])
                    if expected_key[-4:] == ground_truth_key[-4:]:
                    # if expected_key in data and ground_truth_key in data:
                        print(f"Processing wavelength={wavelength} for config={full_config}")
                        y_pred = sigMatNormalize(sigMatFilter(data[expected_key][:]))
                        y_true = sigMatNormalize(sigMatFilter(data[ground_truth_key][:]))
                        
                        metrics = calculate_metrics(y_pred, y_true)
                        results.append((expected_key, wavelength, *metrics.values()))
                    else:
                        print(f"Key not corresponding to correct ground truth: {expected_key} is not the same wavelength as {ground_truth_key}")

            else:
                # SCD/SWFD: Match ground truth based on file_key or configuration
                ground_truth_key = None
                if file_key:
                    # Use the ground_truth mapping for SWFD
                    ground_truth_key = dataset_info["ground_truth"].get(file_key)
                else:
                    # Dynamically resolve ground_truth for SCD
                    for key in dataset_info["ground_truth"]:
                        if key in full_config:
                            ground_truth_key = dataset_info["ground_truth"][key]
                            break

                if not ground_truth_key:
                    print(f"[ERROR] Ground truth key not found for config '{full_config}' and file_key '{file_key}'. Skipping...")
                    return results

                if full_config in data and ground_truth_key in data:
                    print(f"Processing config={full_config}")
                    print(f"Ground truth key: {ground_truth_key}")
                    y_pred = sigMatNormalize(sigMatFilter(data[full_config][:]))
                    y_true = sigMatNormalize(sigMatFilter(data[ground_truth_key][:]))
                    
                    metrics = calculate_metrics(y_pred, y_true)
                    results.append((full_config, *metrics.values()))
                else:
                    print(f"[ERROR] Configuration '{full_config}' or ground truth '{ground_truth_key}' not found in data. Skipping...")


    # Handle MAT files for datasets like mice, phantom, v_phantom
    elif dataset in ["mice", "phantom", "v_phantom"]:
        path = dataset_info["path"]
        gt_file = os.path.join(path, dataset + "_full_recon.mat")
        # print(f"Ground truth file: {gt_file}")
        
        config_file = os.path.join(path, dataset + "_" + config + "_recon.mat")

        if not os.path.isfile(gt_file):
            print(f"[WARNING] Ground truth file not found: {gt_file}. Skipping...")
            return results

        if not os.path.isfile(config_file):
            print(f"[WARNING] Configuration file not found: {config_file}. Skipping...")
            return results

        # Load ground truth and configuration data
        gt_data = load_mat_file(gt_file, dataset_info["ground_truth"])
        config_data = load_mat_file(config_file, full_config)

        print(f"Processing config={full_config}")
        print(f"Ground truth key: {dataset_info['ground_truth']}")
        y_pred = sigMatNormalize(sigMatFilter(config_data))
        y_true = sigMatNormalize(sigMatFilter(gt_data))

        metrics = calculate_metrics(y_pred, y_true)
        results.append((full_config, *metrics.values()))

    return results