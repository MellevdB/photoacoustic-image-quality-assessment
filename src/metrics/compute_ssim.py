from skimage.metrics import structural_similarity as ssim
import numpy as np

def compute_ssim(gt, pred):
    return ssim(gt, pred, data_range=gt.max() - gt.min())

if __name__ == "__main__":
    # Example usage
    gt = np.random.rand(256, 256)  # Replace with ground truth
    pred = gt + np.random.normal(0, 0.1, gt.shape)  # Replace with predicted image
    print(f"SSIM: {compute_ssim(gt, pred):.4f}")