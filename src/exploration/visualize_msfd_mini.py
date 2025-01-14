import matplotlib.pyplot as plt
import h5py
import numpy as np

# File path to MSFD-mini dataset
file_path = "data/OADAT/MSFD/MSFD_multisegment_RawBP-mini.h5"

# Visualize a single sample across multiple wavelengths
def visualize_single_sample_across_wavelengths(file_path):
    # Wavelength datasets to visualize
    wavelengths = [
        "linear_BP_w700", "linear_BP_w730", "linear_BP_w760",
        "linear_BP_w780", "linear_BP_w800", "linear_BP_w850"
    ]
    
    with h5py.File(file_path, "r") as hdf:
        sample_index = 0  # Change this to view a different sample
        plt.figure(figsize=(15, 5))
        for i, wavelength in enumerate(wavelengths):
            data = np.array(hdf[wavelength])
            plt.subplot(1, len(wavelengths), i + 1)
            plt.imshow(data[sample_index], cmap="gray")
            plt.title(f"{wavelength}\nSample {sample_index}")
            plt.axis("off")
        plt.tight_layout()
        plt.show()

# Visualize multiple samples at the same wavelength
def visualize_multiple_samples_at_same_wavelength(file_path):
    dataset_name = "linear_BP_w700"  # Change this to a different wavelength if needed
    
    with h5py.File(file_path, "r") as hdf:
        data = np.array(hdf[dataset_name])
        plt.figure(figsize=(15, 5))
        for i in range(5):  # Visualize first 5 samples
            plt.subplot(1, 5, i + 1)
            plt.imshow(data[i], cmap="gray")
            plt.title(f"Sample {i}")
            plt.axis("off")
        plt.tight_layout()
        plt.show()

# Run visualizations
if __name__ == "__main__":
    print("Visualizing a single sample across multiple wavelengths...")
    visualize_single_sample_across_wavelengths(file_path)
    
    print("Visualizing multiple samples at the same wavelength...")
    visualize_multiple_samples_at_same_wavelength(file_path)