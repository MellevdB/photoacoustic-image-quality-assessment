import matplotlib.pyplot as plt
import h5py
import numpy as np

# File paths to SWFD-mini datasets
file_paths = [
    "/Users/mellevanderbrugge/Documents/UvA/Master AI/Master Thesis/Data (OADAT)/SWFD/SWFD_multisegment_RawBP-mini.h5",
    "/Users/mellevanderbrugge/Documents/UvA/Master AI/Master Thesis/Data (OADAT)/SWFD/SWFD_semicircle_RawBP-mini.h5"
]

# Check data range
def check_data_range(file_path, dataset_name):
    with h5py.File(file_path, "r") as hdf:
        if dataset_name in hdf:
            data = np.array(hdf[dataset_name])
            print(f"Dataset: {dataset_name}")
            print(f"Min value: {data.min()}")
            print(f"Max value: {data.max()}")
            print(f"Mean value: {data.mean()}")
            print(f"Shape: {data.shape}")

# Visualize a single sample from multiple datasets
def visualize_sample_from_datasets(file_path, sample_index=0):
    datasets_to_visualize = [
        "linear_BP", "ms_BP", "sc_BP"
    ]
    
    with h5py.File(file_path, "r") as hdf:
        plt.figure(figsize=(15, 5))
        for i, dataset_name in enumerate(datasets_to_visualize):
            if dataset_name in hdf:
                data = np.array(hdf[dataset_name])
                if data.ndim == 3:  # Assume the dataset has 3D shape
                    # Normalize data for visualization
                    if data.max() > data.min():
                        normalized_data = (data - data.min()) / (data.max() - data.min())
                    else:
                        print(f"{dataset_name} contains only zeros.")
                        normalized_data = data
                    plt.subplot(1, len(datasets_to_visualize), i + 1)
                    plt.imshow(normalized_data[sample_index], cmap="gray")
                    plt.title(f"{dataset_name}\nSample {sample_index}")
                    plt.axis("off")
        plt.tight_layout()
        plt.show()

# Visualize multiple samples from a specific dataset
def visualize_multiple_samples(file_path, dataset_name, num_samples=5):
    with h5py.File(file_path, "r") as hdf:
        if dataset_name in hdf:
            data = np.array(hdf[dataset_name])
            plt.figure(figsize=(15, 5))
            for i in range(min(num_samples, data.shape[0])):
                plt.subplot(1, num_samples, i + 1)
                if data.ndim == 3:  # Assume the dataset has 3D shape
                    # Enhance contrast with percentile clipping
                    if data.max() > data.min():
                        vmin, vmax = np.percentile(data[i], [1, 99])
                        plt.imshow(data[i], cmap="gray", vmin=vmin, vmax=vmax)
                    else:
                        print(f"{dataset_name} Sample {i} contains only zeros.")
                        plt.imshow(data[i], cmap="gray")
                plt.title(f"Sample {i}")
                plt.axis("off")
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    # Loop through both datasets
    for file_path in file_paths:
        print(f"Exploring {file_path}...")

        # Check data range for a key dataset
        print("Checking data range for linear_BP...")
        check_data_range(file_path, "linear_BP")

        # Visualize a single sample across key datasets
        print("Visualizing a single sample across datasets...")
        visualize_sample_from_datasets(file_path, sample_index=0)

        # Visualize multiple samples for a specific dataset
        print("Visualizing multiple samples from linear_BP...")
        visualize_multiple_samples(file_path, dataset_name="linear_BP", num_samples=5)