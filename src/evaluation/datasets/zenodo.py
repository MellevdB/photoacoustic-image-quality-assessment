import os
import cv2
import numpy as np
from evaluation.metrics.calculate import calculate_metrics

def process_zenodo_data(dataset_info, results, metric_type):
    print(dataset_info["path"])
    print(dataset_info["reference"])
    reference_path = os.path.join(dataset_info["path"], dataset_info["reference"])
    algorithms_path = os.path.join(dataset_info["path"], dataset_info["algorithms"])

    reference_images = sorted([f for f in os.listdir(reference_path) if f.endswith(".png")])
    y_true_stack = [cv2.imread(os.path.join(reference_path, f), cv2.IMREAD_GRAYSCALE) for f in reference_images]
    y_true_stack = np.stack(y_true_stack, axis=0)
    print("Stacked reference images:", y_true_stack.shape)

    for category in dataset_info["categories"]:
        y_pred_stack = []
        for ref_img in reference_images:
            number = ref_img.replace("image", "").replace(".png", "")
            pred_img = f"image{number}_{category}.png"
            pred_path = os.path.join(algorithms_path, pred_img)
            if not os.path.exists(pred_path):
                continue
            img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            y_pred_stack.append(img)

        if y_pred_stack:
            y_pred_stack = np.stack(y_pred_stack, axis=0)
            print(f"Stacked predicted images for category {category}:", y_pred_stack.shape)
            metrics = calculate_metrics(y_pred_stack, y_true_stack, metric_type)
            results.append((f"method_{category}", "reference", metrics))