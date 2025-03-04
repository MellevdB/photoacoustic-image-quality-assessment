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

import os
import numpy as np
import cv2
# import torch
# from joblib import Parallel, delayed

def preprocess_bp_data(bp_data):
    """Scale each reconstructed image by its own maximum value and clip all values below -0.2."""
    scaled_data = np.zeros_like(bp_data)
    for i in range(bp_data.shape[0]):  # Process each image slice separately
        max_val = np.max(bp_data[i])
        if max_val > 0:
            scaled_data[i] = bp_data[i] / max_val  # Scale by max value
        scaled_data[i] = np.clip(scaled_data[i], -0.2, None)  # Clip below -0.2
    return scaled_data

def stack_images(image_dir, file_extension=".png", fake_results=False):
    """
    Stacks all images in a directory along the first axis.

    :param image_dir: Path to the directory containing images.
    :param file_extension: The image file format (e.g., .png, .jpg).
    :return: Stacked NumPy array of images.
    """
    if fake_results:
        print(f"FAKE RESULTS: Skipping image loading for {image_dir}")
        return np.random.rand(10, 128, 128)  # Generate dummy data with 10 fake slices

    image_files = sorted([f for f in os.listdir(image_dir) if f.endswith(file_extension)])
    stacked_images = []

    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        if img is None:
            print(f"Warning: Failed to load {img_path}. Skipping...")
            continue
        stacked_images.append(img)

    if not stacked_images:
        print(f"Warning: No images found in {image_dir}. Returning empty array.")
        return np.empty((0, 0, 0))

    return np.stack(stacked_images, axis=0)  # Stack images along first dimension

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
    print("Going to calculate metrics")
    num_images = y_pred.shape[0]
    print(f"Number of images: {num_images}")

    if fake_results:
        print("Using FAKE metric values for quick testing!")
        metrics_mean = {key: np.random.uniform(0.1, 1.0) for key in ['PSNR', 'SSIM', 'VIF', 'FSIM', 'UQI', 'S3IM', 'BRISQUE']}
        metrics_std = {key: np.random.uniform(0.01, 0.1) for key in metrics_mean}
        return metrics_mean, metrics_std

    fr_values = {key: [] for key in ['PSNR', 'SSIM', 'VIF', 'FSIM', 'UQI', 'S3IM']}
    nr_values = {key: [] for key in ['BRISQUE', 'NIQE', 'NIQE-K']}
    
    if metric_type in ["fr", "all"]:
        data_range = y_true.max() - y_true.min()
        
        for i in range(num_images):
            print(f"Calculating FR metrics for image {i}")
            fr_values['PSNR'].append(calculate_psnr(y_true[i], y_pred[i], data_range))
            fr_values['SSIM'].append(calculate_ssim(y_true[i], y_pred[i], data_range))
            fr_values['VIF'].append(calculate_vifp(y_true[i], y_pred[i]))
            fr_values['FSIM'].append(fsim(y_true[i], y_pred[i]))
            fr_values['UQI'].append(calculate_uqi(y_true[i], y_pred[i]))
            fr_values['S3IM'].append(calculate_s3im(y_true[i], y_pred[i]))
        
        # Compute mean and std for FR metrics
        fr_metrics = {key: np.mean(values) for key, values in fr_values.items()}
        fr_std = {key: np.std(values) for key, values in fr_values.items()}
    else:
        fr_metrics, fr_std = {}, {}

    if metric_type in ["nr", "all"]:
        for i in range(num_images):
            print(f"Calculating NR metrics for image {i}")
            nr_values['BRISQUE'].append(calculate_brisque(y_pred[i]))
            # nr_values['NIQE'].append(calculate_niqe(y_pred[i]))
            # nr_values['NIQE-K'].append(calculate_niqe_k(y_pred[i]))

        # Compute mean and std for NR metrics
        nr_metrics = {k: np.mean(v) for k, v in nr_values.items()}
        nr_std = {k: np.std(v) for k, v in nr_values.items()}
    else:
        nr_metrics, nr_std = {}, {}

    # Merge results
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

    if dataset in ["SCD", "SWFD", "MSFD"]:
        _process_hdf5_dataset(dataset, dataset_info, full_config, file_key, results, metric_type, fake_results)
    elif dataset in ["mice", "phantom", "v_phantom"]:
        _process_mat_dataset(dataset, dataset_info, config, full_config, results, metric_type, fake_results)
    elif dataset == "denoising_data":
        _process_denoising_data(dataset, dataset_info, results, metric_type, fake_results)
    elif dataset == "pa_experiment_data":
        _process_pa_experiment_data(dataset, dataset_info, results, metric_type, fake_results)

    return results

def _process_denoising_data(dataset, dataset_info, results, metric_type, fake_results):
    """Processes denoising data (10db - 50db)."""
    base_path = dataset_info["path"]
    subset = "train"  # We are processing train first
    for quality in dataset_info["categories"][:-1]:  # Skip ground_truth for now
        print(f"Processing dataset ={dataset}, config={quality}, ground truth={dataset_info['ground_truth']}")
        quality_path = os.path.join(base_path, "nne", subset, quality)
        ground_truth_path = os.path.join(base_path, "nne", subset, dataset_info["ground_truth"])

        if not os.path.exists(quality_path) or not os.path.exists(ground_truth_path):
            print(f"Skipping {quality}, missing directories.")
            continue

        # Stack images
        
        y_pred = stack_images(quality_path, file_extension=".png", fake_results=fake_results)
        y_true = stack_images(ground_truth_path, file_extension=".png", fake_results=fake_results)
        print(f"Stacked the images")

        if y_pred.shape != y_true.shape:
            print(f"Skipping {quality} due to shape mismatch {y_pred.shape} vs {y_true.shape}")
            continue

        # Compute metrics
        if fake_results:
            y_pred = np.random.rand(128, 128, 128)
            y_true = np.random.rand(128, 128, 128)
        metrics = calculate_metrics(y_pred, y_true, metric_type, fake_results)
        results.append((f"denoising_data/noise/train", quality, "ground_truth", "---", metrics))

def _process_pa_experiment_data(dataset, dataset_info, results, metric_type, fake_results):
    """Processes PA Experiment data (KneeSlice1, Phantoms, SmallAnimal, Transducers) with fixed top-cropping."""
    base_path = dataset_info["path"]
    subset = "Training"

    # Define cropping ratios per dataset
    crop_ratios = {
        "KneeSlice1": 0.20,  # Remove top 20%
        "Phantoms": 0.10,    # Remove top 10%
        "SmallAnimal": 0.10, # Remove top 10%
        "Transducers": 0.10  # Remove top 10%
    }

    for category in dataset_info["training_categories"]:
        category_path = os.path.join(base_path, subset, category)
        if not os.path.exists(category_path):
            print(f"Skipping {category}, missing directory.")
            continue

        subfolders = [f for f in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, f))]

        if not subfolders:
            print(f"No subfolders found in {category_path}. Skipping...")
            continue

        for quality_level in range(2, 8):  # PA2 to PA7
            print(f"Processing dataset={dataset}, category={category}, config=PA{quality_level}, ground truth={dataset_info['ground_truth']}")
            y_pred_stack = []
            y_true_stack = []

            for subfolder in subfolders:
                subfolder_path = os.path.join(category_path, subfolder)
                pred_path = os.path.join(subfolder_path, f"PA{quality_level}.png")
                gt_path = os.path.join(subfolder_path, dataset_info["ground_truth"])

                if not os.path.exists(pred_path) or not os.path.exists(gt_path):
                    print(f"Missing PA{quality_level}.png in {subfolder}. Skipping...")
                    continue

                y_pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
                y_true = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)

                if y_pred is not None and y_true is not None:
                    # Get cropping percentage
                    crop_ratio = crop_ratios.get(category, 0.0)
                    
                    if crop_ratio > 0:
                        h, w = y_pred.shape  # Get height and width
                        crop_height = int(h * crop_ratio)  # Calculate pixels to crop
                        y_pred = y_pred[crop_height:, :]  # Crop top portion
                        y_true = y_true[crop_height:, :]  # Crop top portion
                        print(f"Cropped the images")

                    y_pred_stack.append(y_pred)
                    y_true_stack.append(y_true)

            if y_pred_stack and y_true_stack:
                y_pred_stack = np.stack(y_pred_stack, axis=0)
                y_true_stack = np.stack(y_true_stack, axis=0)
                print(f"Stacked the images")

                # Compute metrics
                if fake_results:
                    y_pred_stack = np.random.rand(128, 128, 128)
                    y_true_stack = np.random.rand(128, 128, 128)
                metrics = calculate_metrics(y_pred_stack, y_true_stack, metric_type, fake_results)
                results.append((f"pa_experiment_data/Training/{category}", f"PA{quality_level}", "PA1", "---", metrics))

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
        print(f"Processing MSFD dataset with config={full_config}, wavelength={wavelength} and ground truth={ground_truth_key}")
        expected_key = f"{full_config}"
        if expected_key[-4:] == ground_truth_key[-4:]:

            if fake_results:
                print("Fake results: ", fake_results)

            y_pred = np.random.rand(128, 128, 128) if fake_results else preprocess_bp_data(data[expected_key][:])
            y_true = np.random.rand(128, 128, 128) if fake_results else preprocess_bp_data(data[ground_truth_key][:])
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
    
    print(f"Processing dataset={dataset}, config={full_config} with ground truth={ground_truth_key}")

    if fake_results:
        print("Fake results: ", fake_results)

    y_pred = np.random.rand(128, 128, 128) if fake_results else preprocess_bp_data(data[full_config][:])
    y_true = np.random.rand(128, 128, 128) if fake_results else preprocess_bp_data(data[ground_truth_key][:])
    metrics = calculate_metrics(y_pred, y_true, metric_type, fake_results)
    results.append((full_config, ground_truth_key, metrics))

def _process_mat_dataset(dataset, dataset_info, config, full_config, results, metric_type, fake_results):
    """Process MAT-file-based datasets (mice, phantom, v_phantom)."""
    path = dataset_info["path"]
    gt_file, config_file = os.path.join(path, f"{dataset}_full_recon.mat"), os.path.join(path, f"{dataset}_{config}_recon.mat")

    if not all(map(os.path.isfile, [gt_file, config_file])):
        print(f"[WARNING] Missing MAT files for {dataset}. Skipping...")
        return

    print(f"Processing dataset={dataset}, config={full_config} with ground truth={gt_file}")

    if fake_results:
        print("Fake results: ", fake_results)

    y_pred = np.random.rand(128, 128, 128) if fake_results else load_mat_file(config_file, full_config)
    y_true = np.random.rand(128, 128, 128) if fake_results else load_mat_file(gt_file, dataset_info["ground_truth"])

    

    metrics = calculate_metrics(y_pred, y_true, metric_type, fake_results)
    results.append((full_config, dataset_info["ground_truth"], metrics))