import os
import h5py

def explore_mouse_data(data_path):
    """
    Explore and summarize the mouse data files.

    :param data_path: Path to the mouse dataset folder.
    """
    print(f"Exploring mouse data in: {data_path}")

    files = [f for f in os.listdir(data_path) if f.endswith('.mat')]
    for file in files:
        file_path = os.path.join(data_path, file)
        print(f"Loading {file}...")
        
        # Load the .mat file using h5py for MATLAB v7.3 format
        with h5py.File(file_path, "r") as data:
            print(f"Keys in {file}: {list(data.keys())}")
            for key in data:
                print(f"Key: {key}, Shape: {data[key].shape}, Type: {data[key].dtype}")
        print("=" * 50)

if __name__ == "__main__":
    mouse_data_path = "data/mice"
    phantom_data_path = "data/phantom"
    v_phantom_data_path = "data/v_phantom"
    print("Exploring mouse data...")
    explore_mouse_data(mouse_data_path)
    print("Exploring phantom data...")
    explore_mouse_data(phantom_data_path)
    print("Exploring v_phantom data...")
    explore_mouse_data(v_phantom_data_path)