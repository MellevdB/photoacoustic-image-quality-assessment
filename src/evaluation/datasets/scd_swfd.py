import h5py
import os
import numpy as np
import imageio
from evaluation.metrics.calculate import calculate_metrics
from evaluation.preprocess.scale_clip import scale_and_clip
from evaluation.preprocess.normalize import min_max_normalize_per_image

def save_original_images(img_array, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(img_array):
        path = os.path.join(output_dir, f"{prefix}_slice_{i}.png")
        # Normalize to [0, 255] just for saving (no clipping!)
        scaled_img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
        imageio.imwrite(path, (scaled_img * 255).astype(np.uint8))

def process_scd_swfd(dataset, dataset_info, full_config, file_key, results, metric_type):
    file_path = dataset_info["path"].get(file_key) if isinstance(dataset_info["path"], dict) else dataset_info["path"]
    if not os.path.isfile(file_path):
        print(f"[WARNING] File not found: {file_path}. Skipping...")
        return
    
    with h5py.File(file_path, 'r') as data:
        ground_truth_key = dataset_info["ground_truth"].get(file_key) if file_key else next(
            (dataset_info["ground_truth"][key] for key in dataset_info["ground_truth"] if key in full_config), None)

        if not ground_truth_key or full_config not in data or ground_truth_key not in data:
            print(f"[ERROR] Missing configuration {full_config} or ground truth {ground_truth_key}. Skipping...")
            return
        
        print(f"Processing dataset={dataset}, config={full_config} with ground truth={ground_truth_key}")

        # save_original_images(data[full_config][:], os.path.join("results", dataset, "original_images"), full_config)

        # y_pred = scale_and_clip(data[full_config][:])
        # y_true = scale_and_clip(data[ground_truth_key][:])
        # y_pred = data[full_config][:]
        # y_true = data[ground_truth_key][:]
        y_pred = min_max_normalize_per_image(data[full_config][:])
        y_true = min_max_normalize_per_image(data[ground_truth_key][:])

        image_ids = [f"{full_config}_slice_{i}" for i in range(y_pred.shape[0])]

        metrics_mean, metrics_std, raw_metrics, image_ids = calculate_metrics(
            y_pred, y_true, metric_type, image_ids=image_ids, store_images=True
            )

        results.append((full_config, ground_truth_key, (metrics_mean, metrics_std, raw_metrics, image_ids)))