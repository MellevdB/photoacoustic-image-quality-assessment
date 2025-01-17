import matplotlib.pyplot as plt
import h5py
import numpy as np

file_path = "data/OADAT/SWFD/SWFD_multisegment_RawBP-mini.h5"

def list_keys(file_path):
    with h5py.File(file_path, "r") as hdf:
        print("Available keys in the file:")
        for key in hdf.keys():
            print(f"- {key}, shape: {hdf[key].shape}")

def visualize_sample_from_datasets(file_path, sample_index=0):
    with h5py.File(file_path, "r") as hdf:
        datasets_to_visualize = [key for key in hdf.keys() if isinstance(hdf[key], h5py.Dataset)]
        print(f"Visualizing datasets: {datasets_to_visualize}")
        plt.figure(figsize=(15, 5))
        for i, dataset_name in enumerate(datasets_to_visualize):
            data = np.array(hdf[dataset_name])
            if data.ndim >= 2:  # Plot only if data is at least 2D
                if np.issubdtype(data.dtype, np.number):
                    normalized_data = (data - data.min()) / (data.max() - data.min()) if data.max() > data.min() else data
                    plt.subplot(1, len(datasets_to_visualize), i + 1)
                    plt.imshow(normalized_data[sample_index], cmap="gray")
                    plt.title(f"{dataset_name}\nSample {sample_index}")
                    plt.axis("off")
                else:
                    print(f"Skipping non-numerical dataset: {dataset_name}")
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    print("Listing all keys...")
    list_keys(file_path)

    # print("\nVisualizing a single sample across all datasets...")
    # visualize_sample_from_datasets(file_path, sample_index=0)