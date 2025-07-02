import os
import pandas as pd
from collections import defaultdict

# Constants
ROOT_DIR = "results/eval_model"
MODEL_FOLDERS = ["best_model", "EfficientNetIQA", "IQDCNN"]
SCD_SUBSETS = ["SCD_ms_ss32", "SCD_ms_ss64", "SCD_ms_ss128"]

# Storage
l1_data = defaultdict(lambda: defaultdict(dict))  # model -> metric -> dataset -> L1
mse_data = defaultdict(lambda: defaultdict(dict))

def parse_loss_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
        l1 = float(lines[0].split(":")[-1])
        mse = float(lines[1].split(":")[-1])
        return l1, mse

def process_model(model_name):
    model_path = os.path.join(ROOT_DIR, model_name)
    for metric in os.listdir(model_path):
        metric_path = os.path.join(model_path, metric)
        if not os.path.isdir(metric_path):
            continue

        for dataset in os.listdir(metric_path):
            if dataset.startswith("SCD_ms_ss") or dataset.startswith("SCD_vc"):
                loss_file = os.path.join(metric_path, dataset, f"test_loss_{dataset}.txt")
            else:
                loss_file = os.path.join(metric_path, dataset, f"test_loss_{dataset}.txt")

            if os.path.exists(loss_file):
                l1, mse = parse_loss_file(loss_file)
                l1_data[model_name][metric][dataset] = l1
                mse_data[model_name][metric][dataset] = mse

def average_scd_losses(metric_dict):
    averaged = {}
    for model, metrics in metric_dict.items():
        for metric, results in metrics.items():
            scd_vals = [results[k] for k in SCD_SUBSETS if k in results]
            if scd_vals:
                avg_scd = sum(scd_vals) / len(scd_vals)
                new_results = {k: v for k, v in results.items() if k not in SCD_SUBSETS}
                new_results["SCD_ms_avg"] = avg_scd
                metric_dict[model][metric] = new_results
    return metric_dict

def to_table(metric_dict):
    rows = []
    for model, metrics in metric_dict.items():
        for metric, datasets in metrics.items():
            row = {"Model": model, "Metric": metric}
            row.update(datasets)
            rows.append(row)
    return pd.DataFrame(rows)

# Run everything
for model in MODEL_FOLDERS:
    process_model(model)

l1_data = average_scd_losses(l1_data)
mse_data = average_scd_losses(mse_data)

l1_df = to_table(l1_data)
mse_df = to_table(mse_data)

# Define actual single metrics (add others here if needed)
SINGLE_METRICS = {"SSIM", "HAARPSI", "IWSSIM", "S3IM", "GMSD_norm"}

def is_combo(metric):
    return metric not in SINGLE_METRICS

l1_single = l1_df[~l1_df["Metric"].apply(is_combo)].sort_values(by=["Model", "Metric"])
l1_combo = l1_df[l1_df["Metric"].apply(is_combo)].sort_values(by=["Model", "Metric"])

mse_single = mse_df[~mse_df["Metric"].apply(is_combo)].sort_values(by=["Model", "Metric"])
mse_combo = mse_df[mse_df["Metric"].apply(is_combo)].sort_values(by=["Model", "Metric"])

# Print tables
print("\n--- L1 Loss: Single Metrics ---")
print(l1_single.to_string(index=False))

print("\n--- L1 Loss: Metric Combinations ---")
print(l1_combo.to_string(index=False))

print("\n--- MSE Loss: Single Metrics ---")
print(mse_single.to_string(index=False))

print("\n--- MSE Loss: Metric Combinations ---")
print(mse_combo.to_string(index=False))

# Save to CSV
output_dir = os.path.join(ROOT_DIR, "table_test_loss")
os.makedirs(output_dir, exist_ok=True)

l1_single.to_csv(os.path.join(output_dir, "test_l1_single_metrics.csv"), index=False)
l1_combo.to_csv(os.path.join(output_dir, "test_l1_combination_metrics.csv"), index=False)

mse_single.to_csv(os.path.join(output_dir, "test_mse_single_metrics.csv"), index=False)
mse_combo.to_csv(os.path.join(output_dir, "test_mse_combination_metrics.csv"), index=False)