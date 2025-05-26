import numpy as np

def min_max_normalize_per_image(batch):
    """
    Normalize each 2D image in a 3D batch array to [0, 1] individually.
    Input:
        batch: np.ndarray of shape (N, H, W)
    Returns:
        normalized_batch: np.ndarray of shape (N, H, W) normalized per image
    """
    batch = batch.astype(np.float32)
    min_vals = np.min(batch, axis=(1, 2), keepdims=True)
    max_vals = np.max(batch, axis=(1, 2), keepdims=True)
    normalized = (batch - min_vals) / (max_vals - min_vals + 1e-8)
    return np.clip(normalized, 0, 1)