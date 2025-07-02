import os

# Define the base directory where the model results are stored
BASE_DIR = "results/eval_model/best_model"

# List of metrics to check
metrics_to_train = [
    'CLIP-IQA', 'SSIM', 'PSNR_norm', 'VIF', 'GMSD_norm', 'HAARPSI',
    'MSSSIM', 'IWSSIM', 'MSGMSD_norm', 'BRISQUE_norm', 'TV', 'UQI', 'S3IM', 'FSIM'
]

# List of datasets
datasets = ['mice', 'MSFD', 'pa_experiment_data', 'phantom', 'v_phantom', 'zenodo']

# Dictionary to store the results
results = {}

# Loop through all metrics and datasets
for metric in metrics_to_train:
    results[metric] = {}
    for dataset in datasets:
        test_loss_path = os.path.join(BASE_DIR, metric, dataset, f"test_loss_{dataset}.txt")
        if os.path.exists(test_loss_path):
            with open(test_loss_path, "r") as f:
                line = f.readline().strip()
                if line.startswith("Test L1 Loss:"):
                    try:
                        loss = float(line.split(":")[-1].strip())
                        results[metric][dataset] = loss
                    except ValueError:
                        results[metric][dataset] = None
                else:
                    results[metric][dataset] = None
        else:
            results[metric][dataset] = None

# Print header with average column
header = ["Metric"] + datasets + ["Avg. Loss"]
print("\t".join(header))

# Print table rows
for metric in metrics_to_train:
    values = [results[metric][ds] for ds in datasets]
    # Filter out missing/invalid values
    valid_values = [v for v in values if isinstance(v, float)]
    avg_loss = sum(valid_values) / len(valid_values) if valid_values else None
    row = [metric] + [f"{v:.4f}" if isinstance(v, float) else "N/A" for v in values]
    row.append(f"{avg_loss:.4f}" if avg_loss is not None else "N/A")
    print("\t".join(row))