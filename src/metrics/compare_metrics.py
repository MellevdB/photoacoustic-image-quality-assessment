import numpy as np
from src.metrics.compute_psnr import compute_psnr
from src.metrics.compute_ssim import compute_ssim
from src.data_loading.extract_data import extract_data

def compare_metrics(file_path, in_key, out_key):
    inputs, ground_truths = extract_data(file_path, in_key, out_key)
    
    psnr_values = []
    ssim_values = []
    
    for pred, gt in zip(inputs, ground_truths):
        psnr_values.append(compute_psnr(gt, pred))
        ssim_values.append(compute_ssim(gt, pred))
    
    return psnr_values, ssim_values

if __name__ == "__main__":
    # Example usage
    file_path = "data/OADAT/SCD_RawBP-mini.h5"
    in_key = "vc,ss32_BP"
    out_key = "vc_BP"
    psnr_vals, ssim_vals = compare_metrics(file_path, in_key, out_key)
    print(f"Average PSNR: {np.mean(psnr_vals):.2f} dB, Average SSIM: {np.mean(ssim_vals):.4f}")