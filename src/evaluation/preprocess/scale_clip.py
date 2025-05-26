import numpy as np

def scale_and_clip(bp_data):
    scaled = bp_data / np.maximum(np.max(bp_data, axis=(1, 2), keepdims=True), 1e-8)
    return np.clip(scaled, 0, None)