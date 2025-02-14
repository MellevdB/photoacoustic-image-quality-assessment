import os
import h5py
import numpy as np

# Define dataset paths
dataset_paths = {
    "SCD": "data/OADAT/SCD/SCD_RawBP-mini.h5",
    "SWFD_multisegment": "data/OADAT/SWFD/SWFD_multisegment_RawBP-mini.h5",
    "SWFD_semicircle": "data/OADAT/SWFD/SWFD_semicircle_RawBP-mini.h5",
    "MSFD": "data/OADAT/MSFD/MSFD_multisegment_RawBP-mini.h5",
}

# Define dataset keys for each dataset
dataset_keys = {
    "SCD": [
        "vc,lv128_BP", "vc,ss128_BP", "vc,ss64_BP", "vc,ss32_BP",
        "linear_BP", "ms,ss128_BP", "ms,ss64_BP", "ms,ss32_BP",
        "vc_BP", "ms_BP"
    ],
    "SWFD_multisegment": [
        "ms,ss128_BP", "ms,ss64_BP", "ms,ss32_BP", "linear_BP", "ms_BP"
    ],
    "SWFD_semicircle": [
        "sc,lv128_BP", "sc,ss128_BP", "sc,ss64_BP", "sc,ss32_BP", "sc_BP"
    ],
    "MSFD": [
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
    print(f"\n Exploring {dataset_name} dataset in: {file_path}")

    if not os.path.exists(file_path):
        print(f"⚠️ Path does not exist: {file_path}")
        return

    with h5py.File(file_path, "r") as data:
        print(f" Available keys in {dataset_name}: {list(data.keys())}")

        for key in keys:
            if key in data:
                dataset = data[key][()]
                
                # Compute statistics
                min_val = np.min(dataset)
                max_val = np.max(dataset)
                mean_val = np.mean(dataset)

                print(f"\nDataset: {key}")
                print(f"   ➤ Shape: {dataset.shape}")
                print(f"   ➤ Type: {dataset.dtype}")
                print(f"   ➤ Min: {min_val:.4f}, Max: {max_val:.4f}, Mean: {mean_val:.4f}")
                print("-" * 50)
            else:
                print(f" Key '{key}' not found in {dataset_name}")

if __name__ == "__main__":
    # Process SCD dataset
    explore_dataset("SCD", dataset_paths["SCD"], dataset_keys["SCD"])

    # Process SWFD datasets separately
    print("\n Exploring SWFD datasets...")
    explore_dataset("SWFD_multisegment", dataset_paths["SWFD_multisegment"], dataset_keys["SWFD_multisegment"])
    explore_dataset("SWFD_semicircle", dataset_paths["SWFD_semicircle"], dataset_keys["SWFD_semicircle"])

    # Process MSFD dataset
    explore_dataset("MSFD", dataset_paths["MSFD"], dataset_keys["MSFD"])