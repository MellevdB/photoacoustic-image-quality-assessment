import os
import h5py
import numpy as np

def explore_mouse_data(data_path):
    """
    Explore and summarize the mouse data files.

    :param data_path: Path to the mouse dataset folder.
    """
    print(f"Exploring mouse data in: {data_path}")

    if not os.path.exists(data_path):
        print(f"⚠️ Path does not exist: {data_path}")
        return

    files = [f for f in os.listdir(data_path) if f.endswith('.mat') or f.endswith('.h5')]
    if not files:
        print(f"⚠️ No .mat or .h5 files found in {data_path}")
        return

    for file in files:
        file_path = os.path.join(data_path, file)
        print(f"\nLoading {file}...")

        # Load the .mat file using h5py for MATLAB v7.3 format
        with h5py.File(file_path, "r") as data:
            print(f"Keys in {file}: {list(data.keys())}")

            for key in data:
                dataset = data[key]
                
                # Check if the dataset is actually an array (not a group)
                if isinstance(dataset, h5py.Dataset):
                    # Read dataset into a NumPy array
                    array_data = dataset[()]
                    
                    # Compute statistics
                    min_val = np.min(array_data)
                    max_val = np.max(array_data)
                    mean_val = np.mean(array_data)

                    print(f"Key: {key}")
                    print(f"   ➤ Shape: {array_data.shape}")
                    print(f"   ➤ Type: {array_data.dtype}")
                    print(f"   ➤ Min: {min_val:.4f}, Max: {max_val:.4f}, Mean: {mean_val:.4f}")
                    print("-" * 50)
        print("=" * 80)

if __name__ == "__main__":
    mouse_data_path = "data/mice"
    phantom_data_path = "data/phantom"
    v_phantom_data_path = "data/v_phantom"

    print("Exploring mouse data...")
    explore_mouse_data(mouse_data_path)

    print("\nExploring phantom data...")
    explore_mouse_data(phantom_data_path)

    print("\nExploring v_phantom data...")
    explore_mouse_data(v_phantom_data_path)