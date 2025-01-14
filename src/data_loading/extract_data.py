import os
import numpy as np
import h5py
from src.data_loading.generators import Generator_Paired_Input_Output

def extract_data(file_path, in_key, out_key, indices=None):
    """
    Extracts sparse reconstructions and corresponding ground truth images.
    
    Args:
        file_path (str): Path to the HDF5 file.
        in_key (str): Key for sparse reconstruction data.
        out_key (str): Key for ground truth data.
        indices (list or None): List of indices to extract, or None for all data.
    
    Returns:
        tuple: (inputs, ground_truths), both as numpy arrays.
    """
    generator = Generator_Paired_Input_Output(
        fname_h5=file_path, in_key=in_key, out_key=out_key, inds=indices
    )
    
    inputs, ground_truths = [], []
    for x, y in generator:
        inputs.append(x)
        ground_truths.append(y)
    
    return np.array(inputs), np.array(ground_truths)

if __name__ == "__main__":
    # Example usage
    file_path = "data/OADAT/SCD_RawBP-mini.h5"
    in_key = "vc,ss32_BP"  # Sparse reconstruction
    out_key = "vc_BP"      # Ground truth
    inputs, ground_truths = extract_data(file_path, in_key, out_key)
    print(f"Extracted {len(inputs)} samples.")