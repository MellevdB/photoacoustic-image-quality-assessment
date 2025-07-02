import os
import h5py
import numpy as np

dataset_paths = {
    "SCD_full": "/projects/prjs1596/photoacoustic/data/OADAT-full/SCD_RawBP.h5",
    "SCD_ms_sparse_full": "/projects/prjs1596/photoacoustic/data/OADAT-full/SCD_multisegment_ss_RawBP.h5",
    "SWFD_multisegment_full": "/projects/prjs1596/photoacoustic/data/OADAT-full/SWFD_multisegment_RawBP.h5",
    "SWFD_multisegment_ss_full": "/projects/prjs1596/photoacoustic/data/OADAT-full/SWFD_multisegment_ss_RawBP.h5",
    "SWFD_semicircle_full": "/projects/prjs1596/photoacoustic/data/OADAT-full/SWFD_semicircle_RawBP.h5",
    "MSFD_full": "/projects/prjs1596/photoacoustic/data/OADAT-full/MSFD_multisegment_RawBP.h5",
    "MSFD_sparse_full": "/projects/prjs1596/photoacoustic/data/OADAT-full/MSFD_multisegment_ss_RawBP.h5",
}

# Keys we expect, can be adjusted later
dataset_keys = {
    "SCD_full": [
        "vc,lv128_BP", "vc,ss128_BP", "vc,ss64_BP", "vc,ss32_BP",
        "linear_BP", "ms,ss128_BP", "ms,ss64_BP", "ms,ss32_BP",
        "vc_BP", "ms_BP"
    ],
    "SWFD_multisegment_full": [
        "ms,ss128_BP", "ms,ss64_BP", "ms,ss32_BP", "linear_BP", "ms_BP"
    ],
    "SWFD_semicircle_full": [
        "sc,lv128_BP", "sc,ss128_BP", "sc,ss64_BP", "sc,ss32_BP", "sc_BP"
    ],
    "MSFD_full": [
        "ms,ss32_BP_w700", "ms,ss32_BP_w730", "ms,ss32_BP_w760", "ms,ss32_BP_w780",
        "ms,ss32_BP_w800", "ms,ss32_BP_w850", "ms,ss64_BP_w700", "ms,ss64_BP_w730",
        "ms,ss64_BP_w760", "ms,ss64_BP_w780", "ms,ss64_BP_w800", "ms,ss64_BP_w850",
        "ms,ss128_BP_w700", "ms,ss128_BP_w730", "ms,ss128_BP_w760", "ms,ss128_BP_w780",
        "ms,ss128_BP_w800", "ms,ss128_BP_w850",
        "ms_BP_w700", "ms_BP_w730", "ms_BP_w760", "ms_BP_w780", "ms_BP_w800", "ms_BP_w850"
    ],
}

def explore_dataset(dataset_name, file_path, keys):
    """
    Explore and summarize a given dataset.

    :param dataset_name: Name of the dataset (e.g., "SCD", "MSFD").
    :param file_path: Path to the dataset file.
    :param keys: List of keys to analyze in the dataset.
    """
    print(f"\nExploring {dataset_name} at: {file_path}")

    if not os.path.exists(file_path):
        print(f"Path does not exist: {file_path}")
        return

    with h5py.File(file_path, "r") as data:
        available_keys = list(data.keys())
        print(f"Available keys in {dataset_name}:")
        for k in available_keys:
            print(f"  - {k}")

        # for key in keys:
        #     if key in data:
        #         dataset = data[key][()]
        #         min_val = np.min(dataset)
        #         max_val = np.max(dataset)
        #         mean_val = np.mean(dataset)

        #         print(f"\nFound: {key}")
        #         print(f"   Shape: {dataset.shape}")
        #         print(f"   Type: {dataset.dtype}")
        #         print(f"   Min: {min_val:.4f}, Max: {max_val:.4f}, Mean: {mean_val:.4f}")
        #         print("-" * 50)
        #     else:
        #         print(f"Key not found: {key}")

if __name__ == "__main__":
    explore_dataset("SCD_full", dataset_paths["SCD_full"], dataset_keys.get("SCD_full", []))
    explore_dataset("SCD_ms_sparse_full", dataset_paths["SCD_ms_sparse_full"], dataset_keys.get("SCD_ms_sparse_full", []))
    explore_dataset("SWFD_multisegment_full", dataset_paths["SWFD_multisegment_full"], dataset_keys.get("SWFD_multisegment_full", []))
    # explore_dataset("SWFD_multisegment_ss_full", dataset_paths["SWFD_multisegment_ss_full"], dataset_keys.get("SWFD_multisegment_ss_full", []))
    # explore_dataset("SWFD_semicircle_full", dataset_paths["SWFD_semicircle_full"], dataset_keys.get("SWFD_semicircle_full", []))
    # explore_dataset("MSFD_full", dataset_paths["MSFD_full"], dataset_keys.get("MSFD_full", []))
    # explore_dataset("MSFD_sparse_full", dataset_paths["MSFD_sparse_full"], dataset_keys.get("MSFD_sparse_full", []))