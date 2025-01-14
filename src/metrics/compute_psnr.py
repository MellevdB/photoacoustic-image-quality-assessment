import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr

def compute_psnr(gt, pred):
    return psnr(gt, pred, data_range=gt.max() - gt.min())

if __name__ == "__main__":
    # Example usage
    gt = np.random.rand(256, 256)  # Replace with ground truth
    pred = gt + np.random.normal(0, 0.1, gt.shape)  # Replace with predicted image
    print(f"PSNR: {compute_psnr(gt, pred):.2f} dB")