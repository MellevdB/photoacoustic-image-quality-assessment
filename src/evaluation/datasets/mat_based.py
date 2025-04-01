import os
from evaluation.metrics.calculate import calculate_metrics
import h5py
from scipy.io import loadmat

def load_mat_file(file_path, key):
    try:
        # Try loading with scipy (for older MATLAB formats)
        return loadmat(file_path)[key]
    except NotImplementedError:
        # Use h5py for MATLAB v7.3 files
        with h5py.File(file_path, 'r') as f:
            return f[key][()]

def process_mat_dataset(dataset, dataset_info, config, full_config, results, metric_type):
    path = dataset_info["path"]
    gt_path = os.path.join(path, f"{dataset}_full_recon.mat")
    pred_path = os.path.join(path, f"{dataset}_{config}_recon.mat")
    if os.path.isfile(gt_path) and os.path.isfile(pred_path):
        y_true = load_mat_file(gt_path, dataset_info["ground_truth"])
        y_pred = load_mat_file(pred_path, full_config)
        metrics = calculate_metrics(y_pred, y_true, metric_type)
        results.append((full_config, dataset_info["ground_truth"], metrics))