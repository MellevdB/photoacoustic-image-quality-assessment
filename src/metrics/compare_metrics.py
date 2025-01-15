import argparse
import numpy as np
from src.metrics.metrics import compute_psnr
from src.metrics.metrics import compute_ssim
from src.data_loading_oadat.extract_data_oadat import extract_data


def compare_metrics(file_path, in_key, out_key):
    inputs, ground_truths = extract_data(file_path, in_key, out_key)

    # Print shapes for debugging
    for i, (inp, gt) in enumerate(zip(inputs, ground_truths)):
        print(f"Sample {i}: Input shape {inp.shape}, Ground truth shape {gt.shape}")

    psnr_values = [compute_psnr(gt, pred) for gt, pred in zip(ground_truths, inputs)]
    ssim_values = [compute_ssim(gt, pred) for gt, pred in zip(ground_truths, inputs)]
    return psnr_values, ssim_values

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare PSNR and SSIM metrics.")
    parser.add_argument("--file_path", required=True, help="Path to the HDF5 file.")
    parser.add_argument("--in_key", required=True, help="Key for sparse reconstruction data.")
    parser.add_argument("--out_key", required=True, help="Key for ground truth data.")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to evaluate.")
    
    args = parser.parse_args()
    psnr_vals, ssim_vals = compare_metrics(args.file_path, args.in_key, args.out_key)
    print(f"PSNR (first {args.num_samples} samples): {psnr_vals[:args.num_samples]}")
    print(f"SSIM (first {args.num_samples} samples): {ssim_vals[:args.num_samples]}")

# Example usage:
# python src/metrics/compare_metrics.py --file_path data/OADAT/SCD/SCD_RawBP-mini.h5 --in_key vc,ss32_BP --out_key vc_BP --num_samples 5