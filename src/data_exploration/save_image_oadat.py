import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

DATA_DIR = "data/OADAT"
RESULTS_DIR = "results/OADAT"

# Dataset configuration
DATASETS = {
    "SCD": {
        "path": os.path.join(DATA_DIR, "SCD/SCD_RawBP-mini.h5"),
        "configs": {
            "vc": ["vc,lv128_BP", "vc,ss128_BP", "vc,ss64_BP", "vc,ss32_BP"],
            "linear": ["linear_BP"],
            "ms": ["ms,ss128_BP", "ms,ss64_BP", "ms,ss32_BP"],
        },
        "ground_truth": {
            "vc": "vc_BP",
            "linear": "linear_BP",
            "ms": "ms_BP",
        },
    },
    "SWFD": {
        "path": {
            "multisegment": os.path.join(DATA_DIR, "SWFD/SWFD_multisegment_RawBP-mini.h5"),
            "semicircle": os.path.join(DATA_DIR, "SWFD/SWFD_semicircle_RawBP-mini.h5"),
        },
        "configs": {
            "multisegment": ["linear_BP", "ms,ss128_BP", "ms,ss64_BP", "ms,ss32_BP"],
            "semicircle": ["sc,lv128_BP", "sc,ss128_BP", "sc,ss64_BP", "sc,ss32_BP"],
        },
        "ground_truth": {
            "multisegment": "ms_BP",
            "semicircle": "sc_BP",
        },
    },
    "MSFD": {
        "path": os.path.join(DATA_DIR, "MSFD/MSFD_multisegment_RawBP-mini.h5"),
        "configs": {
            "ss32": ["ms,ss32_BP_w700", "ms,ss32_BP_w730", "ms,ss32_BP_w760", "ms,ss32_BP_w780", "ms,ss32_BP_w800", "ms,ss32_BP_w850"],
            "ss64": ["ms,ss64_BP_w700", "ms,ss64_BP_w730", "ms,ss64_BP_w760", "ms,ss64_BP_w780", "ms,ss64_BP_w800", "ms,ss64_BP_w850"],
            "ss128": ["ms,ss128_BP_w700", "ms,ss128_BP_w730", "ms,ss128_BP_w760", "ms,ss128_BP_w780", "ms,ss128_BP_w800", "ms,ss128_BP_w850"],
        },
        "ground_truth": {
            "wavelengths": {
                700: "ms_BP_w700",
                730: "ms_BP_w730",
                760: "ms_BP_w760",
                780: "ms_BP_w780",
                800: "ms_BP_w800",
                850: "ms_BP_w850",
            },
        },
    },
}

def save_image(data, save_path_prefix):
    """Save the given 2D or 3D data as images."""
    if data.ndim == 2:
        plt.figure(figsize=(6, 6))
        plt.imshow(data, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path_prefix + ".png", bbox_inches="tight", pad_inches=0)
        plt.close()
    elif data.ndim == 3:
        num_slices = data.shape[0]
        indices = np.linspace(0, num_slices - 1, 5, dtype=int)
        for i, slice_idx in enumerate(indices):
            slice_data = data[slice_idx, :, :]
            save_path = f"{save_path_prefix}_slice{i + 1}.png"
            plt.figure(figsize=(6, 6))
            plt.imshow(slice_data, cmap="gray")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
            plt.close()

def process_dataset(dataset_name, dataset_info):
    """Process a single dataset and save images."""
    print(f"Processing dataset: {dataset_name}")

    if isinstance(dataset_info["path"], dict):
        # For datasets with multiple file keys (e.g., SWFD)
        for file_key, file_path in dataset_info["path"].items():
            print(f"Opening file: {file_path} (key: {file_key})")
            with h5py.File(file_path, "r") as data:
                print(f"Keys in dataset: {list(data.keys())}")
                for config, keys in dataset_info["configs"].items():
                    for key in keys:
                        if key in data:
                            print(f"Processing key: {key} in config: {config}")
                            save_path_prefix = os.path.join(RESULTS_DIR, dataset_name, file_key, config, key)
                            os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)
                            save_image(data[key][()], save_path_prefix)
                        else:
                            print(f"Key '{key}' not found in dataset '{file_key}'")

                # Ground truth
                gt_key = dataset_info["ground_truth"].get(file_key)
                if gt_key:
                    if gt_key in data:
                        print(f"Processing ground truth: {gt_key}")
                        save_path_prefix = os.path.join(RESULTS_DIR, dataset_name, file_key, "ground_truth")
                        os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)
                        save_image(data[gt_key][()], save_path_prefix)
                    else:
                        print(f"Ground truth key '{gt_key}' not found in dataset '{file_key}'")
    else:
        # For datasets with a single file path (e.g., SCD, MSFD)
        print(f"Opening file: {dataset_info['path']}")
        with h5py.File(dataset_info["path"], "r") as data:
            print(f"Keys in dataset: {list(data.keys())}")
            for config, keys in dataset_info["configs"].items():
                for key in keys:
                    if key in data:
                        print(f"Processing key: {key} in config: {config}")
                        save_path_prefix = os.path.join(RESULTS_DIR, dataset_name, config, key)
                        os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)
                        save_image(data[key][()], save_path_prefix)
                    else:
                        print(f"Key '{key}' not found in dataset '{dataset_name}'")

            # Ground truth
            if dataset_name == "MSFD":
                # Handle MSFD wavelengths
                gt_keys = dataset_info["ground_truth"]["wavelengths"]
                for wavelength, gt_key in gt_keys.items():
                    if gt_key in data:
                        print(f"Processing ground truth: {gt_key} (wavelength: {wavelength})")
                        save_path_prefix = os.path.join(RESULTS_DIR, dataset_name, "ground_truth", f"{gt_key}_{wavelength}")
                        os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)
                        save_image(data[gt_key][()], save_path_prefix)
                    else:
                        print(f"Ground truth key '{gt_key}' not found in dataset '{dataset_name}'")
            else:
                # Handle other datasets
                gt_keys = dataset_info["ground_truth"]
                for gt_key in gt_keys.values():
                    if gt_key in data:
                        print(f"Processing ground truth: {gt_key}")
                        save_path_prefix = os.path.join(RESULTS_DIR, dataset_name, "ground_truth", gt_key)
                        os.makedirs(os.path.dirname(save_path_prefix), exist_ok=True)
                        save_image(data[gt_key][()], save_path_prefix)
                    else:
                        print(f"Ground truth key '{gt_key}' not found in dataset '{dataset_name}'")

def main():
    for dataset_name, dataset_info in DATASETS.items():
        process_dataset(dataset_name, dataset_info)

if __name__ == "__main__":
    main()