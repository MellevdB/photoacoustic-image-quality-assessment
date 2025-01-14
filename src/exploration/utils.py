import h5py
import numpy as np

def explore_hdf5(file_path):
    """Prints the structure and shape of datasets in an HDF5 file."""
    with h5py.File(file_path, "r") as hdf:
        print(f"Exploring {file_path}")
        for key in hdf.keys():
            print(f"Top-level key: {key}")
            if isinstance(hdf[key], h5py.Group):
                print(f"{key} is a group containing: {list(hdf[key].keys())}")
            elif isinstance(hdf[key], h5py.Dataset):
                print(f"{key} is a dataset with shape {hdf[key].shape} and dtype {hdf[key].dtype}")

def load_dataset(file_path, group_name, dataset_name):
    """Loads a specific dataset from an HDF5 file."""
    with h5py.File(file_path, "r") as hdf:
        data = np.array(hdf[group_name][dataset_name])
    return data