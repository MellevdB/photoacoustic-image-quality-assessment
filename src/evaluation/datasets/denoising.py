import os
from evaluation.metrics.calculate import calculate_metrics
from evaluation.preprocess.image_loader import load_image_stack

def process_denoising_data(dataset_info, results, metric_type):
    base = dataset_info["path"]
    subset = "train"
    for quality in dataset_info["categories"][:-1]:
        pred_path = os.path.join(base, "nne", subset, quality)
        gt_path = os.path.join(base, "nne", subset, dataset_info["ground_truth"])
        y_pred = load_image_stack(pred_path)
        y_true = load_image_stack(gt_path)
        if y_pred.shape == y_true.shape:
            metrics = calculate_metrics(y_pred, y_true, metric_type)
            results.append((f"denoising_data/noise/train", quality, "ground_truth", "---", metrics))