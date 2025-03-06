import os
import numpy as np
import matplotlib.pyplot as plt

# File paths to results
result_files = {
    "SCD": "results/SCD/SCD_results_2025-03-04_20-27-50.txt",
    "MSFD": "results/MSFD/MSFD_results_2025-03-04_20-27-52.txt",
    "SWFD": "results/SWFD/SWFD_results_2025-03-04_20-29-37.txt",
}

# Define metric headers
metric_headers = ["FSIM", "UQI", "PSNR", "SSIM", "VIF", "S3IM"]
nr_metrics = ["BRISQUE"]  # No-reference metrics

# Dataset-specific configurations
dataset_configs = {
    "SCD": {
        "vc": ["vc,ss32_BP", "vc,ss64_BP", "vc,ss128_BP"],
        "ms": ["ms,ss32_BP", "ms,ss64_BP", "ms,ss128_BP"],
    },
    "MSFD": {
        "msfd": ["ms,ss32_BP_w760", "ms,ss64_BP_w760", "ms,ss128_BP_w760"],
    },
    "SWFD": {
        "sc": ["sc,ss32_BP", "sc,ss64_BP", "sc,ss128_BP"],
        "ms": ["ms,ss32_BP", "ms,ss64_BP", "ms,ss128_BP"],
    },
}

# --- Function to load results ---
def load_results(file_path):
    """Reads a results file and extracts dataset configurations and metric values."""
    if not os.path.exists(file_path):
        print(f"Skipping {file_path}: File not found.")
        return None

    with open(file_path, "r") as f:
        lines = f.readlines()

    # Extract header and clean up
    header = lines[0].strip().split("   ")
    data_lines = lines[2:]  # Skip header and separator line

    results = []
    for line in data_lines:
        values = line.strip().split("   ")
        dataset, config, gt, wavelength = values[:4]
        metrics = [float(v) if v.replace(".", "", 1).isdigit() else None for v in values[4:]]
        results.append((dataset, config, gt, wavelength, metrics))

    return header, results


# --- Function to organize data per dataset ---
def organize_data(results, dataset_name):
    """Organizes data into a structured dictionary for plotting."""
    dataset_data = {group: {metric: {"mean": [], "std": []} for metric in metric_headers + nr_metrics}
                    for group in dataset_configs[dataset_name]}

    for dataset, config, gt, wavelength, metrics in results:
        for group, configs in dataset_configs[dataset_name].items():
            if config in configs:
                metric_idx = 4
                for metric in metric_headers + nr_metrics:
                    mean_idx = metric_idx
                    std_idx = metric_idx + 1
                    if mean_idx < len(metrics) and std_idx < len(metrics):
                        dataset_data[group][metric]["mean"].append(metrics[mean_idx])
                        dataset_data[group][metric]["std"].append(metrics[std_idx])
                    metric_idx += 2  # Move to next metric pair

    return dataset_data


# --- Function to plot metrics with y-axis rupture ---
def plot_metrics(configs, metric_means, metric_stds, title):
    """Creates a line plot for multiple metrics with a ruptured y-axis."""
    x = np.arange(len(configs))

    # Create two subplots: one for PSNR, one for other metrics
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 1]})

    # Define colors
    colors = {
        "FSIM": "b",
        "UQI": "g",
        "PSNR": "r",
        "SSIM": "c",
        "VIF": "m",
        "S3IM": "y"
    }

    # Loop over each metric
    for metric, mean_values in metric_means.items():
        std_values = metric_stds[metric]
        color = colors.get(metric, "k")

        if metric == "PSNR":
            ax_top.plot(x, mean_values, label=metric, color=color, linestyle="dashed", marker="o")
            ax_top.fill_between(x, np.array(mean_values) - np.array(std_values), np.array(mean_values) + np.array(std_values), color=color, alpha=0.2)
        else:
            ax_bottom.plot(x, mean_values, label=metric, color=color, marker="o")
            ax_bottom.fill_between(x, np.array(mean_values) - np.array(std_values), np.array(mean_values) + np.array(std_values), color=color, alpha=0.2)

    # Formatting
    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)
    ax_top.set_ylabel("PSNR (dB)", color="r", fontsize=18)
    ax_bottom.set_ylabel("Metric Value", fontsize=18)
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(configs, rotation=45, fontsize=16)
    ax_top.legend(loc="upper right", fontsize=14)
    ax_bottom.legend(loc="upper left", fontsize=14)
    plt.suptitle(title, fontsize=20)
    plt.show()


# --- Process and plot data for each dataset ---
for dataset_name, file_path in result_files.items():
    result_data = load_results(file_path)
    if result_data is None:
        continue

    header, results = result_data
    organized_data = organize_data(results, dataset_name)

    for group, data in organized_data.items():
        configs = dataset_configs[dataset_name][group]

        # Prepare metric data
        metric_data = {metric: {"mean": [], "std": []} for metric in metric_headers}
        for metric in metric_headers:
            metric_data[metric]["mean"] = [np.mean(data[metric]["mean"]) if data[metric]["mean"] else 0]
            metric_data[metric]["std"] = [np.mean(data[metric]["std"]) if data[metric]["std"] else 0]

        # Plot full-reference metrics
        plot_metrics(configs,
                     {metric: metric_data[metric]["mean"] for metric in metric_headers},
                     {metric: metric_data[metric]["std"] for metric in metric_headers},
                     f"{dataset_name} - {group}")

        # Prepare no-reference metric data
        nr_metric_data = {metric: {"mean": [], "std": []} for metric in nr_metrics}
        for metric in nr_metrics:
            nr_metric_data[metric]["mean"] = [np.mean(data[metric]["mean"]) if data[metric]["mean"] else 0]
            nr_metric_data[metric]["std"] = [np.mean(data[metric]["std"]) if data[metric]["std"] else 0]

        # Plot no-reference metrics
        if any(nr_metric_data[metric]["mean"] for metric in nr_metrics):
            plot_metrics(configs,
                         {metric: nr_metric_data[metric]["mean"] for metric in nr_metrics},
                         {metric: nr_metric_data[metric]["std"] for metric in nr_metrics},
                         f"{dataset_name} - {group} (No-Reference)")