import os
import scipy.io as sio

def explore_v_phantom_data(data_path):
    """
    Explore and summarize the virtual phantom data files.

    :param data_path: Path to the v_phantom dataset folder.
    """
    print(f"Exploring virtual phantom data in: {data_path}")

    files = [f for f in os.listdir(data_path) if f.endswith('.mat')]
    for file in files:
        file_path = os.path.join(data_path, file)
        print(f"Loading {file}...")
        
        # Load the .mat file
        data = sio.loadmat(file_path)
        
        # Explore the keys and data
        print(f"Keys in {file}: {list(data.keys())}")
        for key in data:
            if not key.startswith("__"):  # Skip meta keys
                print(f"Key: {key}, Shape: {data[key].shape}, Type: {type(data[key])}")
        print("=" * 50)

if __name__ == "__main__":
    v_phantom_data_path = "data/v_phantom/9250634"
    explore_v_phantom_data(v_phantom_data_path)