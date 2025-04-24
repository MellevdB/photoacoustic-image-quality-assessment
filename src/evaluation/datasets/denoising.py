import os
from evaluation.metrics.calculate import calculate_metrics
from evaluation.preprocess.image_loader import load_image_stack_with_ids

def process_denoising_data(dataset_info, results, metric_type):
    base = dataset_info["path"]
    subset = "train"
    for quality in dataset_info["categories"][:-1]:
        pred_path = os.path.join(base, "nne", subset, quality)
        gt_path = os.path.join(base, "nne", subset, dataset_info["ground_truth"])

        y_pred, image_ids = load_image_stack_with_ids(pred_path)
        y_true, _ = load_image_stack_with_ids(gt_path)  # same file names assumed

        if y_pred.shape == y_true.shape:
            metrics_mean, metrics_std, raw_metrics, _ = calculate_metrics(y_pred, y_true, metric_type, image_ids=image_ids, store_images=True)
            results.append((f"denoising_data/noise/train", quality, "ground_truth", "---", (metrics_mean, metrics_std, raw_metrics, image_ids)))