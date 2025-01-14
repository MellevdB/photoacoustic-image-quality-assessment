import argparse
import numpy as np
from src.data_loading_oadat.generators import Generator_Paired_Input_Output
from src.preprocessing_oadat.preprocess_oadat import preprocess_data

def extract_data(file_path, in_key, out_key, indices=None, apply_filter=True):
    """
    Extract and preprocess data from OADAT dataset.

    Args:
        file_path (str): Path to the HDF5 file.
        in_key (str): Key for sparse reconstruction data.
        out_key (str): Key for ground truth data.
        indices (list or None): Indices to extract, or None for all.
        apply_filter (bool): Whether to apply bandpass filtering during preprocessing.

    Returns:
        tuple: (inputs, ground_truths) as numpy arrays.
    """
    generator = Generator_Paired_Input_Output(
        fname_h5=file_path, in_key=in_key, out_key=out_key, inds=indices
    )
    inputs, ground_truths = [], []
    
    for x, y in generator:
        # Preprocess the input (sparse data)
        x_preprocessed = preprocess_data(x, apply_filter=apply_filter)
        inputs.append(x_preprocessed)
        ground_truths.append(y)  # Ground truth data doesn't need preprocessing
    
    return np.array(inputs), np.array(ground_truths)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and preprocess data from OADAT dataset.")
    parser.add_argument("--file_path", required=True, help="Path to the HDF5 file.")
    parser.add_argument("--in_key", required=True, help="Key for sparse reconstruction data.")
    parser.add_argument("--out_key", required=True, help="Key for ground truth data.")
    parser.add_argument("--indices", nargs="+", type=int, help="Indices to extract, or leave empty for all data.")
    parser.add_argument("--no_filter", action="store_true", help="Disable bandpass filtering during preprocessing.")
    
    args = parser.parse_args()
    inputs, ground_truths = extract_data(
        args.file_path, args.in_key, args.out_key, args.indices, not args.no_filter
    )
    print(f"Extracted and preprocessed {len(inputs)} samples.")

# Example usage:
# python src/data_loading_oadat/extract_data_oadat.py --file_path data/OADAT/SCD/SCD_RawBP-mini.h5 --in_key vc,ss32_BP --out_key vc_BP