import os
import cv2
import numpy as np
from evaluation.metrics.calculate import calculate_metrics

def process_pa_experiment_data(dataset_info, results, metric_type):
    base = dataset_info["path"]
    subset = "Training"
    crop = {"KneeSlice1": 0.2, "Phantoms": 0.1, "SmallAnimal": 0.1, "Transducers": 0.1}
    for category in dataset_info["training_categories"]:
        cat_path = os.path.join(base, subset, category)
        if not os.path.isdir(cat_path): continue
        subfolders = [f for f in os.listdir(cat_path) if os.path.isdir(os.path.join(cat_path, f))]
        for q in range(2, 8):
            y_pred, y_true = [], []
            for sub in subfolders:
                p = os.path.join(cat_path, sub)
                pred = cv2.imread(os.path.join(p, f"PA{q}.png"), cv2.IMREAD_GRAYSCALE)
                gt = cv2.imread(os.path.join(p, dataset_info["ground_truth"]), cv2.IMREAD_GRAYSCALE)
                if pred is not None and gt is not None:
                    ch = int(pred.shape[0] * crop.get(category, 0))
                    y_pred.append(pred[ch:])
                    y_true.append(gt[ch:])
            if y_pred:
                y_pred = np.stack(y_pred, axis=0)
                y_true = np.stack(y_true, axis=0)
                metrics = calculate_metrics(y_pred, y_true, metric_type)
                results.append((f"pa_experiment_data/Training/{category}", f"PA{q}", "PA1", "---", metrics))