import os
import h5py
import numpy as np
import pandas as pd
import scipy.io

# Define datasets and their configurations
datasets = {
    "SCD": {
        "file": "data/OADAT/SCD/SCD_RawBP-mini.h5",
        "keys": ['linear_BP', 'linear_raw', 'ms,lv128_BP', 'ms,ss128_BP', 'ms,ss32_BP', 'ms,ss32_raw',
                 'ms,ss64_BP', 'ms,ss64_raw', 'ms_BP', 'ms_raw', 'vc,lv128_BP', 'vc,lv128_raw', 'vc,ss128_BP',
                 'vc,ss128_raw', 'vc,ss32_BP', 'vc,ss32_raw', 'vc,ss64_BP', 'vc,ss64_raw', 'vc_BP', 'vc_raw']
    },
    "SWFD_multisegment": {
        "file": "data/OADAT/SWFD/SWFD_multisegment_RawBP-mini.h5",
        "keys": ['linear_BP', 'linear_raw', 'ms,ss128_BP', 'ms,ss128_raw', 'ms,ss32_BP', 'ms,ss32_raw',
                 'ms,ss64_BP', 'ms,ss64_raw', 'ms_BP', 'ms_raw']
    },
    "SWFD_semicircle": {
        "file": "data/OADAT/SWFD/SWFD_semicircle_RawBP-mini.h5",
        "keys": ['sc,lv128_BP', 'sc,lv128_raw', 'sc,ss128_BP', 'sc,ss128_raw', 'sc,ss32_BP', 'sc,ss32_raw',
                 'sc,ss64_BP', 'sc,ss64_raw', 'sc_BP', 'sc_raw']
    },
    "MSFD": {
        "file": "data/OADAT/MSFD/MSFD_multisegment_RawBP-mini.h5",
        "keys": ['linear_BP_w700', 'linear_BP_w730', 'linear_BP_w760', 'linear_BP_w780', 'linear_BP_w800', 'linear_BP_w850',
                 'linear_raw_w700', 'linear_raw_w730', 'linear_raw_w760', 'linear_raw_w780', 'linear_raw_w800', 'linear_raw_w850',
                 'ms,ss128_BP_w700', 'ms,ss128_BP_w730', 'ms,ss128_BP_w760', 'ms,ss128_BP_w780', 'ms,ss128_BP_w800', 'ms,ss128_BP_w850',
                 'ms,ss128_raw_w700', 'ms,ss128_raw_w730', 'ms,ss128_raw_w760', 'ms,ss128_raw_w780', 'ms,ss128_raw_w800', 'ms,ss128_raw_w850',
                 'ms,ss32_BP_w700', 'ms,ss32_BP_w730', 'ms,ss32_BP_w760', 'ms,ss32_BP_w780', 'ms,ss32_BP_w800', 'ms,ss32_BP_w850',
                 'ms,ss32_raw_w700', 'ms,ss32_raw_w730', 'ms,ss32_raw_w760', 'ms,ss32_raw_w780', 'ms,ss32_raw_w800', 'ms,ss32_raw_w850',
                 'ms,ss64_BP_w700', 'ms,ss64_BP_w730', 'ms,ss64_BP_w760', 'ms,ss64_BP_w780', 'ms,ss64_BP_w800', 'ms,ss64_BP_w850',
                 'ms,ss64_raw_w700', 'ms,ss64_raw_w730', 'ms,ss64_raw_w760', 'ms,ss64_raw_w780', 'ms,ss64_raw_w800', 'ms,ss64_raw_w850']
    },
    "Mice": {
        "folder": "data/mice/",
        "files": ["mice_full_recon.mat", "mice_sparse4_recon.mat", "mice_sparse8_recon.mat", "mice_sparse16_recon.mat",
                  "mice_sparse32_recon.mat", "mice_sparse64_recon.mat", "mice_sparse128_recon.mat", "mice_sparse256_recon.mat"]
    },
    "Phantom": {
        "folder": "data/phantom/",
        "files": ["phantom_full_recon.mat", "phantom_sparse8_recon.mat", "phantom_sparse16_recon.mat",
                  "phantom_sparse32_recon.mat", "phantom_sparse64_recon.mat", "phantom_sparse128_recon.mat"]
    },
    "V_Phantom": {
        "folder": "data/v_phantom/",
        "files": ["v_phantom_full_recon.mat", "v_phantom_sparse8_recon.mat", "v_phantom_sparse16_recon.mat",
                  "v_phantom_sparse32_recon.mat", "v_phantom_sparse64_recon.mat", "v_phantom_sparse128_recon.mat"]
    }
}

# List to store dataset statistics
stats = []

# Function to extract statistics from HDF5 files
def process_hdf5(file_path, dataset_name, key):
    with h5py.File(file_path, "r") as f:
        if key in f:
            data = f[key][()]
            stats.append([dataset_name, key, data.shape, data.dtype, np.min(data), np.max(data), np.mean(data), np.std(data)])
        else:
            print(f"⚠ Key {key} not found in {file_path}")

# Function to extract statistics from MATLAB (.mat) files
def process_matlab(file_path, dataset_name):
    try:
        with h5py.File(file_path, "r") as f:  # MATLAB v7.3 files (HDF5)
            for key in f.keys():
                data = f[key][()]
                stats.append([dataset_name, key, data.shape, data.dtype, np.min(data), np.max(data), np.mean(data), np.std(data)])
    except OSError:
        # Try scipy.io for older MATLAB versions
        try:
            mat_data = scipy.io.loadmat(file_path)
            for key, data in mat_data.items():
                if isinstance(data, np.ndarray):
                    stats.append([dataset_name, key, data.shape, data.dtype, np.min(data), np.max(data), np.mean(data), np.std(data)])
        except Exception as e:
            print(f"Error processing {dataset_name} - {file_path}: {e}")

# Process HDF5 datasets
for dataset_name, info in datasets.items():
    if "file" in info:  # HDF5 files
        file_path = info["file"]
        if os.path.exists(file_path):
            for key in info["keys"]:
                try:
                    process_hdf5(file_path, dataset_name, key)
                except Exception as e:
                    print(f"Error processing {dataset_name} - {key}: {e}")

# Process MATLAB datasets (v7.3 HDF5 format)
for dataset_name, info in datasets.items():
    if "folder" in info:  # MATLAB files
        folder_path = info["folder"]
        for file_name in info["files"]:
            file_path = os.path.join(folder_path, file_name)
            if os.path.exists(file_path):
                try:
                    process_matlab(file_path, dataset_name)
                except Exception as e:
                    print(f"Error processing {dataset_name} - {file_name}: {e}")

# Save results to CSV
df = pd.DataFrame(stats, columns=["Dataset", "Key", "Shape", "Dtype", "Min", "Max", "Mean", "Std"])
csv_file = "dataset_statistics.csv"
df.to_csv(csv_file, index=False)

print(f"✅ Dataset statistics saved to {csv_file}")