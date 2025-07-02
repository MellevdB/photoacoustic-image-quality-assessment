import os
import sys
import numpy as np
import h5py
import scipy.io as sio
import cv2
import glob
import matplotlib.pyplot as plt
from scipy.fftpack import fft


# Base directories
DATA_DIR = "data/"
RESULTS_DIR = "results/"

# Load dataset config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../config")))

from data_config import DATASETS


def check_h5_dataset(dataset_name, dataset_info):
    """Check if HDF5 dataset contains raw signals or preprocessed images efficiently."""
    paths = dataset_info["path"]

    if isinstance(paths, dict):  # Multi-file datasets (e.g., SWFD)
        for sub_dataset, path in paths.items():
            print(f"Checking {dataset_name} - {sub_dataset} at {path}")
            process_h5_file(dataset_name, path, dataset_info["configs"], sub_dataset)
    else:
        print(f"Checking {dataset_name} at {paths}")
        process_h5_file(dataset_name, paths, dataset_info["configs"], None)


def process_h5_file(dataset_name, path, configs, sub_dataset=None):
    """Process a single HDF5 file and extract dataset information, including raw vs BP comparisons."""
    if not os.path.exists(path):
        print(f"Error: File not found - {path}")
        return

    try:
        with h5py.File(path, "r", libver="latest", swmr=True) as f:
            available_keys = list(f.keys())
            dataset_label = f"{dataset_name} ({sub_dataset})" if sub_dataset else dataset_name
            print(f"  Available keys in {dataset_label}: {available_keys}")

            # Ensure we only check relevant keys
            valid_configs = {config: keys for config, keys in configs.items() if any(k in available_keys for k in keys)}

            for config, keys in valid_configs.items():
                for key in keys:
                    if key not in available_keys:
                        continue  # Skip keys that do not exist

                    dset = f[key]
                    print(f"  {config} - {key}: shape {dset.shape}, dtype {dset.dtype}")

                    # Check if corresponding `_raw` version exists
                    raw_key = key.replace("_BP", "_raw")  # Convert key name to expected raw format
                    has_raw = raw_key in available_keys

                    # Compute stats over the first 10 samples instead of just 1
                    num_samples = min(10, dset.shape[0])  # Use at most 10 samples if available
                    sample_bp = np.array(dset[:num_samples])

                    min_bp, max_bp = sample_bp.min(), sample_bp.max()
                    mean_bp, std_bp = sample_bp.mean(), sample_bp.std()

                    print(f"  BP Stats -> Min: {min_bp}, Max: {max_bp}, Mean: {mean_bp:.2f}, Std: {std_bp:.2f}")

                    # If raw data exists, compute its stats
                    if has_raw:
                        dset_raw = f[raw_key]
                        sample_raw = np.array(dset_raw[:num_samples])

                        min_raw, max_raw = sample_raw.min(), sample_raw.max()
                        mean_raw, std_raw = sample_raw.mean(), sample_raw.std()

                        print(f"  RAW Stats ({raw_key}) -> Min: {min_raw}, Max: {max_raw}, Mean: {mean_raw:.2f}, Std: {std_raw:.2f}")

                        # Compare `_BP` vs `_raw`
                        diff_min = min_bp - min_raw
                        diff_max = max_bp - max_raw
                        diff_mean = mean_bp - mean_raw
                        diff_std = std_bp - std_raw

                        print(f"  Difference (BP - RAW) -> Min: {diff_min}, Max: {diff_max}, Mean: {diff_mean:.2f}, Std: {diff_std:.2f}")

                    # Perform FFT check if 2D image data (use a random slice)
                    if sample_bp.ndim == 3:  # Shape (num_samples, height, width)
                        print(f"  FFT Analysis for BP ({key}):")
                        fft_analysis(sample_bp[0], dataset_name, config)

                    if has_raw and sample_raw.ndim == 3:
                        print(f"  FFT Analysis for RAW ({raw_key}):")
                        fft_analysis(sample_raw[0], dataset_name, config)

    except Exception as e:
        print(f"Error reading {dataset_label} from {path}: {e}")


def check_mat_dataset(dataset_name, dataset_info):
    """Check if MATLAB dataset contains raw signals or preprocessed images."""
    path = dataset_info["path"]
    print(f"Checking MATLAB dataset: {dataset_name}")

    mat_files = glob.glob(os.path.join(path, "*.mat"))

    for file in mat_files:
        print(f"Processing: {file}")  # Print before loading

        try:
            mat_data = sio.loadmat(file)
            print(f"Loaded {file} using SciPy")
        except NotImplementedError:
            print(f"{file} is a MATLAB v7.3 file. Using h5py instead.")
            try:
                with h5py.File(file, "r") as f:
                    print(f"  Keys in {file}: {list(f.keys())[:5]}")
                    for key in list(f.keys())[:5]:
                        dset = f[key]
                        sample = dset[0]
                        print(f"  {key}: shape {dset.shape}, dtype {dset.dtype}")
                        print(f"  Stats -> Min: {sample.min()}, Max: {sample.max()}, Mean: {sample.mean():.2f}, Std: {sample.std():.2f}")
                        if np.all(sample >= 0) and np.all(sample <= 1):
                            print(f"  {key} appears to be normalized. Skipping mean normalization.")
            except Exception as e:
                print(f"Error reading {file}: {e}")
                continue


def check_image_dataset(dataset_name, dataset_info):
    """Check image datasets in multiple directories, handling both grayscale and color images."""
    print("In check_image_dataset")
    paths = dataset_info["path"]
    if isinstance(paths, str):
        paths = [paths]

    # Ensure all relevant subdirectories are checked
    if dataset_name == "denoising_data":
        paths.extend(glob.glob("data/denoising_data/drive/train/*db"))
        paths.extend(glob.glob("data/denoising_data/nne/train/*db"))
    elif dataset_name == "pa_experiment_data":
        paths.extend(glob.glob("data/pa_experiment_data/Training/*"))

    print(f"Checking Image dataset: {dataset_name} in {len(paths)} directories.")

    image_files = []
    for path in paths:
        if os.path.exists(path):
            found_images = glob.glob(os.path.join(path, "**", "*.png"), recursive=True) + \
                           glob.glob(os.path.join(path, "**", "*.jpg"), recursive=True) + \
                           glob.glob(os.path.join(path, "**", "*.jpeg"), recursive=True) + \
                           glob.glob(os.path.join(path, "**", "*.tiff"), recursive=True)
            image_files.extend(found_images)
        else:
            print(f"Error: Directory {path} not found.")

    if not image_files:
        print(f"No images found in {dataset_name}.")
        return

    for img_file in image_files[:15]:  # Limit to 15 images for quick analysis
        img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)  # Load with original depth (grayscale or color)

        if img is None:
            print(f"Unable to read {img_file}. Skipping.")
            continue

        # **Handle Grayscale (2D) and Color (3D) Images**
        if len(img.shape) == 2:  # Grayscale
            print(f"  {img_file}: shape {img.shape}, min {img.min()}, max {img.max()}, mean {img.mean():.2f}, std {img.std():.2f}")
            if np.all(img >= 0) and np.all(img <= 1):
                print(f"  {img_file} appears to be normalized. Skipping mean normalization.")

        elif len(img.shape) == 3:  # Color (3D), shape = (H, W, C)
            h, w, c = img.shape
            print(f"  {img_file}: shape ({h}, {w}, {c}) - Color Image (RGB or BGR)")

            # Process each color channel separately
            for channel_idx in range(c):
                channel_data = img[:, :, channel_idx]
                min_val, max_val = channel_data.min(), channel_data.max()
                mean_val, std_val = channel_data.mean(), channel_data.std()

                print(f"    🔵 Channel {channel_idx} -> Min: {min_val}, Max: {max_val}, Mean: {mean_val:.2f}, Std: {std_val:.2f}")

                if np.all(channel_data >= 0) and np.all(channel_data <= 1):
                    print(f"     Channel {channel_idx} appears to be normalized. Skipping mean normalization.")

        else:
            print(f"Unexpected image shape {img.shape} in {img_file}. Skipping.")


def fft_analysis(data, dataset_name, config_name):
    """Perform FFT and print numerical statistics."""
    sample = data[:, 0] if data.ndim == 2 else data
    fft_result = np.abs(fft(sample))
    freq_bins = np.fft.fftfreq(len(sample))

    peak_freq = freq_bins[np.argmax(fft_result)]
    lowest_freq = freq_bins[np.argmin(fft_result)]
    mean_freq = np.mean(fft_result)
    std_freq = np.std(fft_result)

    print(f"FFT Analysis for {dataset_name} - {config_name}:")
    print(f"   Peak Frequency: {peak_freq:.4f}")
    print(f"   Lowest Frequency: {lowest_freq:.4f}")
    print(f"   Mean Frequency: {mean_freq:.4f}")
    print(f"   Std Deviation: {std_freq:.4f}")


if __name__ == "__main__":
    for dataset_name, dataset_info in DATASETS.items():
        dataset_path = dataset_info["path"]

        if isinstance(dataset_path, str) and dataset_path.endswith(".h5"):
            check_h5_dataset(dataset_name, dataset_info)
        elif isinstance(dataset_path, dict):  # Multi-file datasets (e.g., SWFD)
            check_h5_dataset(dataset_name, dataset_info)
        elif dataset_name in ["denoising_data", "pa_experiment_data"]:
            check_image_dataset(dataset_name, dataset_info)
        elif os.path.isdir(dataset_path):  # MATLAB .mat datasets
            if dataset_name in ["mice", "phantom", "v_phantom"]:
                check_mat_dataset(dataset_name, dataset_info)
        