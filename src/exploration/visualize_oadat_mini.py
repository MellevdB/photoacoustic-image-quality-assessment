import matplotlib.pyplot as plt
import h5py
import numpy as np

# File path to OADAT-mini dataset
file_path = "/Users/mellevanderbrugge/Documents/UvA/Master AI/Master Thesis/Data (OADAT)/OADAT-mini/OADAT-mini.h5"

# Check data range
def check_data_range(file_path, group, dataset):
    with h5py.File(file_path, "r") as hdf:
        if group in hdf and dataset in hdf[group]:
            data = np.array(hdf[group][dataset])
            print(f"Dataset: {group}/{dataset}")
            print(f"Min value: {data.min()}")
            print(f"Max value: {data.max()}")
            print(f"Mean value: {data.mean()}")
            print(f"Shape: {data.shape}")

# Visualize a single sample across all groups
def visualize_sample_across_groups(file_path, sample_index=0):
    groups = ["MSFD", "SCD", "SWFD_ms", "SWFD_sc"]
    datasets_to_visualize = {
        "MSFD": ["linear_BP_w700", "linear_BP_w730"],
        "SCD": ["linear_BP", "ground_truth"],
        "SWFD_ms": ["linear_BP"],
        "SWFD_sc": ["sc_BP"]
    }
    
    with h5py.File(file_path, "r") as hdf:
        for group in groups:
            if group in hdf:
                print(f"Visualizing {group} group...")
                plt.figure(figsize=(12, 6))
                datasets = datasets_to_visualize[group]
                for i, dataset_name in enumerate(datasets):
                    if dataset_name in hdf[group]:
                        data = np.array(hdf[group][dataset_name])
                        if data.ndim == 3:  # Assume the dataset has 3D shape
                            # Handle zero data gracefully
                            if data.max() > data.min():
                                normalized_data = (data - data.min()) / (data.max() - data.min())
                            else:
                                print(f"{group}/{dataset_name} contains only zeros.")
                                normalized_data = data
                            plt.subplot(1, len(datasets), i + 1)
                            plt.imshow(normalized_data[sample_index], cmap="gray")
                            plt.title(f"{group} - {dataset_name}\nSample {sample_index}")
                            plt.axis("off")
                plt.tight_layout()
                plt.show()

# Visualize multiple samples from a specific group and dataset
def visualize_multiple_samples(file_path, group, dataset, num_samples=5):
    with h5py.File(file_path, "r") as hdf:
        if group in hdf and dataset in hdf[group]:
            data = np.array(hdf[group][dataset])
            plt.figure(figsize=(15, 5))
            for i in range(min(num_samples, data.shape[0])):
                plt.subplot(1, num_samples, i + 1)
                if data.ndim == 3:  # Assume the dataset has 3D shape
                    # Enhance contrast with percentile clipping
                    if data.max() > data.min():
                        vmin, vmax = np.percentile(data[i], [1, 99])
                        plt.imshow(data[i], cmap="gray", vmin=vmin, vmax=vmax)
                    else:
                        print(f"{group}/{dataset} Sample {i} contains only zeros.")
                        plt.imshow(data[i], cmap="gray")
                plt.title(f"Sample {i}")
                plt.axis("off")
            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    # Check data range
    print("Checking data range for MSFD/linear_BP_w700...")
    check_data_range(file_path, "MSFD", "linear_BP_w700")

    print("Visualizing a single sample across groups...")
    visualize_sample_across_groups(file_path, sample_index=0)

    print("Visualizing multiple samples from MSFD - linear_BP_w700...")
    visualize_multiple_samples(file_path, group="MSFD", dataset="linear_BP_w700", num_samples=5)