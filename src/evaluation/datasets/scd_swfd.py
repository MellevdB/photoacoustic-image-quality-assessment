import h5py
import os
from evaluation.metrics.calculate import calculate_metrics
from evaluation.preprocess.scale_clip import scale_and_clip

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

        y_pred = scale_and_clip(data[full_config][:])
        y_true = scale_and_clip(data[ground_truth_key][:])
        metrics_mean, metrics_std, _ = calculate_metrics(y_pred, y_true, metric_type)
        results.append((full_config, ground_truth_key, (metrics_mean, metrics_std)))