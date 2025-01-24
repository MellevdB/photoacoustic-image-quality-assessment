import os
import h5py
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np

# Define paths for the datasets
DATA_PATH = "data"
RESULTS_PATH = "results"

# Define datasets and their corresponding sparse keys
DATASETS = {
    "mice": {
        "path": os.path.join(DATA_PATH, "mice"),
        "keys": ["sparse4", "sparse8", "sparse16", "sparse32", "sparse64", "sparse128", "sparse256"],
        "gt": "full_recon"
    },
    "phantom": {
        "path": os.path.join(DATA_PATH, "phantom"),
        "keys": ["sparse8", "sparse16", "sparse32", "sparse64", "sparse128"],
        "gt": "full_recon"
    },
    "v_phantom": {
        "path": os.path.join(DATA_PATH, "v_phantom"),
        "keys": ["sparse8", "sparse16", "sparse32", "sparse64", "sparse128"],
        "gt": "full_recon"
    }
}


def save_image(data, save_path_prefix):
    """
    Save the given 2D or 3D data as images.
    - If data is 2D, save it directly.
    - If data is 3D, save 5 evenly spaced slices.

    :param data: Input image data (2D or 3D).
    :param save_path_prefix: Path prefix to save the image(s).
    """
    if data.ndim == 2:
        # Save single 2D image
        plt.figure(figsize=(6, 6))
        plt.imshow(data, cmap="gray")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path_prefix + ".png", bbox_inches="tight", pad_inches=0)
        plt.close()
    elif data.ndim == 3:
        # Save 5 evenly spaced slices for 3D data
        num_slices = data.shape[0]
        indices = np.linspace(0, num_slices - 1, 5, dtype=int)
        print(f"Data is 3D with shape {data.shape}. Saving 5 slices at indices {indices}.")
        
        for i, slice_idx in enumerate(indices):
            slice_data = data[slice_idx, :, :]
            save_path = f"{save_path_prefix}_slice{i + 1}.png"
            plt.figure(figsize=(6, 6))
            plt.imshow(slice_data, cmap="gray")
            plt.axis("off")
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
            plt.close()
    else:
        print(f"Unsupported data shape {data.shape} for saving as an image.")

def load_mat_file(file_path, key=None):
    """
    Load .mat file (supports both regular and v7.3 files).
    """
    try:
        if h5py.is_hdf5(file_path):
            with h5py.File(file_path, "r") as f:
                if key:
                    return np.array(f[key])
                else:
                    # Automatically handle keys
                    return {k: np.array(f[k]) for k in f.keys()}
        else:
            mat_data = sio.loadmat(file_path)
            if key:
                return mat_data[key]
            else:
                return {k: v for k, v in mat_data.items() if not k.startswith("__")}
    except Exception as e:
        print(f"Error loading file {file_path}: {e}")
        return None

def process_dataset(dataset_name, dataset_info):
    """
    Process and save images for a given dataset.
    """
    dataset_path = dataset_info["path"]
    save_dir = os.path.join(RESULTS_PATH, dataset_name)
    os.makedirs(save_dir, exist_ok=True)

    # Process ground truth
    gt_file = os.path.join(dataset_path, f"{dataset_name}_{dataset_info['gt']}.mat")
    if os.path.exists(gt_file):
        print(f"Processing GT for {dataset_name}...")
        data = load_mat_file(gt_file)
        if data:
            for key, value in data.items():
                save_path = os.path.join(save_dir, f"{dataset_name}_gt.png")
                save_image(value, save_path)

    # Process sparse keys
    for key in dataset_info["keys"]:
        sparse_file = os.path.join(dataset_path, f"{dataset_name}_{key}_recon.mat")
        if os.path.exists(sparse_file):
            print(f"Processing {key} for {dataset_name}...")
            data = load_mat_file(sparse_file)
            if data:
                for data_key, value in data.items():
                    save_path = os.path.join(save_dir, f"{dataset_name}_{key}.png")
                    save_image(value, save_path)

def main():
    for dataset_name, dataset_info in DATASETS.items():
        process_dataset(dataset_name, dataset_info)
    print("Image saving completed.")

if __name__ == "__main__":
    main()