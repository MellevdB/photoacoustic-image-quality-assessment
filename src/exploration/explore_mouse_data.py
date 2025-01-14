import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

def explore_mouse_data(file_path):
    """
    Explore and visualize the contents of a mouse dataset file.
    
    Args:
        file_path (str): Path to the HDF5 file.
    """
    with h5py.File(file_path, 'r') as f:
        print(f"Exploring file: {file_path}")
        print("Keys in the dataset:")
        print(list(f.keys()))
        
        # Explore the first key
        first_key = list(f.keys())[0]
        data = f[first_key][:]
        print(f"Shape of data under '{first_key}': {data.shape}")
        
        # Plot the first slice if the data is 2D or 3D
        if data.ndim >= 2:
            plt.imshow(data[0], cmap='gray')
            plt.title(f"First slice of '{first_key}'")
            plt.colorbar()
            plt.show()

if __name__ == "__main__":
    # Define file paths
    base_path = "/Users/mellevanderbrugge/Documents/UvA/Master AI/Master Thesis/photoacoustic-image-quality-assessment/data/mouse"
    file_paths = [
        os.path.join(base_path, "16841759"),
        os.path.join(base_path, "16842449"),
    ]
    
    for file_path in file_paths:
        explore_mouse_data(file_path)