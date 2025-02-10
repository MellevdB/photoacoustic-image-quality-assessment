import matplotlib.pyplot as plt
import h5py
import numpy as np

# File path to SCD-mini dataset
file_path = "data/OADAT/SCD/SCD_RawBP-mini.h5"

# Check available keys in the HDF5 file
def list_keys(file_path):
    with h5py.File(file_path, "r") as hdf:
        print("Available keys in the file:")
        for key in hdf.keys():
            print(f"- {key}, shape: {hdf[key].shape}")

# Check data range for a specific key
def check_data_range(file_path, dataset_name):
    with h5py.File(file_path, "r") as hdf:
        if dataset_name in hdf:
            data = np.array(hdf[dataset_name])
            print(f"Dataset: {dataset_name}")
            print(f"Min value: {data.min()}")
            print(f"Max value: {data.max()}")
            print(f"Mean value: {data.mean()}")
            print(f"Shape: {data.shape}")
        else:
            print(f"Dataset {dataset_name} not found in the file.")

# Visualize a single sample from multiple datasets
def visualize_sample_from_datasets(file_path, sample_index=0):
    with h5py.File(file_path, "r") as hdf:
        datasets_to_visualize = list(hdf.keys())
        print(f"Visualizing datasets: {datasets_to_visualize}")
        plt.figure(figsize=(15, 5))
        for i, dataset_name in enumerate(datasets_to_visualize):
            data = np.array(hdf[dataset_name])
            if data.ndim >= 2:  # Plot only if data is at least 2D
                normalized_data = (data - data.min()) / (data.max() - data.min()) if data.max() > data.min() else data
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
            print(f"Visualizing {num_samples} samples from {dataset_name}")
            plt.figure(figsize=(15, 5))
            for i in range(min(num_samples, data.shape[0])):
                plt.subplot(1, num_samples, i + 1)
                vmin, vmax = np.percentile(data[i], [1, 99]) if data.max() > data.min() else (data.min(), data.max())
                plt.imshow(data[i], cmap="gray", vmin=vmin, vmax=vmax)
                plt.title(f"Sample {i}")
                plt.axis("off")
            plt.tight_layout()
            plt.show()
        else:
            print(f"Dataset {dataset_name} not found in the file.")

def compare_dataset_shapes(file_path, key1, key2):
    with h5py.File(file_path, "r") as hdf:
        if key1 in hdf and key2 in hdf:
            data1, data2 = np.array(hdf[key1]), np.array(hdf[key2])
            print(f"Shapes of {key1}: {data1.shape}, {key2}: {data2.shape}")
        else:
            print(f"One or both keys not found: {key1}, {key2}")

def check_statistics(file_path, dataset_names):
    with h5py.File(file_path, "r") as hdf:
        for dataset_name in dataset_names:
            if dataset_name in hdf:
                data = np.array(hdf[dataset_name])
                print(f"Dataset: {dataset_name}")
                print(f"Min: {data.min()}, Max: {data.max()}, Mean: {data.mean()}")
            else:
                print(f"{dataset_name} not found.")

if __name__ == "__main__":
    # List all available keys in the dataset
    print("Listing all keys...")
    list_keys(file_path)

    # Check data range for a specific dataset
    print("\nChecking data range for ground_truth...")
    check_data_range(file_path, "ground_truth")

    print("Comparing shapes for key pairs...")
    compare_dataset_shapes(file_path, "vc,ss32_BP", "ground_truth")
    compare_dataset_shapes(file_path, "ms,ss32_BP", "ground_truth")

    print("\nChecking statistics for key datasets...")
    check_statistics(file_path, ["vc,ss32_BP", "ground_truth", "ms,ss32_BP", "linear_BP", "vc,ss64_BP", "ms,ss64_BP", "vc,ss128_BP", "ms,ss128_BP", "ms,lv128_BP", "ms_BP", "vc_BP"])

    # # Visualize a single sample across all datasets
    # print("\nVisualizing a single sample across all datasets...")
    # visualize_sample_from_datasets(file_path, sample_index=0)

    # # Visualize multiple samples from a specific dataset
    # print("\nVisualizing multiple samples from linear_BP...")
    # visualize_multiple_samples(file_path, dataset_name="linear_BP", num_samples=5)