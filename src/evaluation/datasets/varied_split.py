import os
import cv2
import numpy as np
import pandas as pd
from evaluation.metrics.piq_metrics import compute_piq_metrics


def process_varied_split(dataset_info, results, metric_type="all", use_csv=True):
    root = dataset_info["path"]
    expert_gt_path = os.path.join(root, "ground_truth_results.csv")

    expert_gt_map = {}
    if use_csv:
        expert_gt_df = pd.read_csv(expert_gt_path)
        expert_gt_map = {
            str(row["scene"]).strip(): str(row["ground_truth"]).strip()
            for _, row in expert_gt_df.iterrows()
        }

    scene_folders = [f for f in os.listdir(root) if f.startswith("scene_")]

    for folder in sorted(scene_folders):
        folder_path = os.path.join(root, folder)
        image_files = [f for f in os.listdir(folder_path) if f.endswith(".webp")]

        # === Use expert GT if available ===
        if use_csv and folder in expert_gt_map:
            expert_gt_substr = expert_gt_map[folder]
            gt_matches = [f for f in image_files if expert_gt_substr in f]
            if len(gt_matches) != 1:
                print(f"[WARNING] Could not uniquely match expert GT in {folder}: '{expert_gt_substr}' -> {gt_matches}")
                continue
            ground_truth = gt_matches[0]
            print(f"[INFO] Expert GT used for {folder}: {ground_truth}")
        else:
            # === Fallback logic (non-expert scenes) ===
            ground_truth = None
            if folder.startswith("scene_22") or folder.startswith("scene_23"):
                ground_truth = [f for f in image_files if "full" in f][0]
            elif folder.startswith("scene_32"):
                ground_truth = [f for f in image_files if f.startswith("SWFD_semicircle_sc_BP")][0]
            elif folder.startswith("scene_33"):
                ground_truth = [f for f in image_files if f.startswith("SWFD_multisegment_ms_BP")][0]
            elif folder.startswith("scene_500"):
                ground_truth = [f for f in image_files if f.endswith("PA_GT.webp")][0]
            elif folder.startswith("scene_510"):
                ground_truth = [f for f in image_files if f.endswith("1.webp")][0]
            elif folder == "scene_6001":
                ground_truth = "Bladder 50-80.webp"
            elif folder == "scene_6002":
                ground_truth = "Brain 51-80.webp"
            elif folder == "scene_6003":
                ground_truth = "Hindlimb 49-80.webp"
            elif folder == "scene_6004":
                ground_truth = "Kidneys cross section -50-80.webp"
            elif folder.startswith("scene_600"):
                ground_truth_candidates = [f for f in image_files if f.endswith("51-80.webp")]
                if len(ground_truth_candidates) != 1:
                    print(f"[WARNING] Unexpected GT candidates in {folder}: {ground_truth_candidates}")
                    continue
                ground_truth = ground_truth_candidates[0]
            else:
                print(f"[SKIP] Unknown pattern in {folder}")
                continue

        predictions = sorted([f for f in image_files if f != ground_truth])

        print(f"[INFO] Folder: {folder}")
        print(f"[INFO] Selected GT: {ground_truth}")
        print(f"[INFO] Prediction files: {predictions}")

        # === Load images ===
        gt_path = os.path.join(folder_path, ground_truth)
        gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if gt_img is None:
            print(f"[ERROR] Failed to load GT image: {gt_path}")
            continue
        gt_img = gt_img / 255.0

        y_true = [gt_img]  # Include GT vs GT
        y_pred = [gt_img]
        image_ids = [gt_path]

        for pred_file in predictions:
            pred_path = os.path.join(folder_path, pred_file)
            pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            if pred_img is None:
                print(f"[SKIP] Failed to load prediction: {pred_path}")
                continue
            pred_img = pred_img / 255.0

            if pred_img.shape != gt_img.shape:
                print(f"[SKIP] Shape mismatch in {pred_file} (expected {gt_img.shape}, got {pred_img.shape})")
                continue

            y_true.append(gt_img)
            y_pred.append(pred_img)
            image_ids.append(pred_path)

        if not y_pred:
            print(f"[SKIP] No valid predictions in {folder}")
            continue

        y_true = np.stack(y_true)
        y_pred = np.stack(y_pred)

        config = os.path.basename(folder_path)
        gt_label = ground_truth

        metrics_mean, metrics_std, raw_metrics, image_ids = compute_piq_metrics(
            y_pred, y_true, image_ids=image_ids, batch_size=64
        )

        results.append((folder_path, config, gt_label, "---", (metrics_mean, metrics_std, raw_metrics, image_ids)))