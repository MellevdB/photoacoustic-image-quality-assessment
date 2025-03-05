import os
import numpy as np
import matplotlib.pyplot as plt

# File paths to results
result_files = {
    "Denoising Data": "results/denoising_data/denoising_data_results_2025-03-04_20-26-49.txt",
    "Mice": "results/mice/mice_results_2025-03-04_20-30-18.txt",
    "MSFD": "results/MSFD/MSFD_results_2025-03-04_20-27-52.txt",
    "PA Experiment": "results/pa_experiment_data/pa_experiment_data_results_2025-03-04_20-27-18.txt",
    "Phantom": "results/phantom/phantom_results_2025-03-04_20-30-35.txt",
    "SCD": "results/SCD/SCD_results_2025-03-04_20-27-50.txt",
    "SWFD": "results/SWFD/SWFD_results_2025-03-04_20-29-37.txt",
    "V Phantom": "results/v_phantom/v_phantom_results_2025-03-04_20-31-30.txt",
}

# Define metric headers
metric_headers = ["FSIM", "UQI", "PSNR", "SSIM", "VIF", "S3IM"]
nr_metrics = ["BRISQUE", "NIQE", "NIQE-K"]


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


def organize_data(results, header):
    """Organizes data into a structured dictionary for plotting."""
    dataset_data = {}

    for dataset, config, gt, wavelength, metrics in results:
        if dataset not in dataset_data:
            dataset_data[dataset] = {}

        if config not in dataset_data[dataset]:
            dataset_data[dataset][config] = {metric: {"mean": [], "std": []} for metric in metric_headers + nr_metrics}

        # Assign metric values
        metric_idx = 4
        for metric in metric_headers + nr_metrics:
            mean_idx = metric_idx
            std_idx = metric_idx + 1
            if mean_idx < len(metrics) and std_idx < len(metrics):
                dataset_data[dataset][config][metric]["mean"].append(metrics[mean_idx])
                dataset_data[dataset][config][metric]["std"].append(metrics[std_idx])
            metric_idx += 2  # Move to next metric pair

    return dataset_data


def plot_metrics(dataset_name, configs, metric_data):
    """
    Creates a line plot for multiple metrics with a ruptured y-axis.

    :param dataset_name: Name of the dataset
    :param configs: List of configurations (x-axis labels)
    :param metric_data: Dictionary containing metric means and stds
    """
    x = np.arange(len(configs))

    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 1]})
    plt.rcParams.update({'font.size': 14})

    for metric, values in metric_data.items():
        means, stds = np.array(values["mean"]), np.array(values["std"])

        if metric == "PSNR":
            ax_top.plot(x, means, label=metric, linestyle="--", marker="o", color="tab:blue")
            ax_top.fill_between(x, means - stds, means + stds, color="tab:blue", alpha=0.2)
        else:
            ax_bottom.plot(x, means, label=metric, linestyle="-", marker="o")
            ax_bottom.fill_between(x, means - stds, means + stds, alpha=0.2)

    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)
    ax_top.set_ylabel("PSNR (dB)", color="tab:blue", fontsize=18)
    ax_bottom.set_ylabel("Metric Value", fontsize=18)
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(configs, rotation=45, fontsize=16)
    ax_top.legend(loc="upper right", fontsize=16)
    ax_bottom.legend(loc="upper left", fontsize=16)
    plt.suptitle(f"{dataset_name} Dataset", fontsize=20)
    plt.show()


def plot_nr_metrics(dataset_name, configs, metric_data):
    """
    Creates a line plot for no-reference metrics.

    :param dataset_name: Name of the dataset
    :param configs: List of configurations (x-axis labels)
    :param metric_data: Dictionary containing NR metric means and stds
    """
    x = np.arange(len(configs))

    plt.figure(figsize=(10, 6))
    for metric, values in metric_data.items():
        means, stds = np.array(values["mean"]), np.array(values["std"])
        plt.plot(x, means, label=metric, linestyle="-", marker="o")
        plt.fill_between(x, means - stds, means + stds, alpha=0.2)

    plt.ylabel("NR Metric Score", fontsize=18)
    plt.xticks(x, configs, rotation=45, fontsize=14)
    plt.title(f"{dataset_name} - No Reference Metrics", fontsize=20)
    plt.legend(fontsize=16)
    plt.grid(True)
    plt.show()


# --- Main Execution ---
for dataset_name, file_path in result_files.items():
    # Load results
    result_data = load_results(file_path)
    if result_data is None:
        continue  # Skip if file is missing

    header, results = result_data
    organized_data = organize_data(results, header)

    for dataset, data in organized_data.items():
        configs = list(data.keys())  # Extract configuration names

        # Prepare data for FR metrics
        metric_data = {metric: {"mean": [], "std": []} for metric in metric_headers}
        for config in configs:
            for metric in metric_headers:
                metric_data[metric]["mean"].append(np.mean(data[config][metric]["mean"]) if data[config][metric]["mean"] else 0)
                metric_data[metric]["std"].append(np.mean(data[config][metric]["std"]) if data[config][metric]["std"] else 0)

        # Plot full-reference metrics
        plot_metrics(dataset, configs, metric_data)

        # Prepare data for NR metrics
        nr_metric_data = {metric: {"mean": [], "std": []} for metric in nr_metrics}
        for config in configs:
            for metric in nr_metrics:
                nr_metric_data[metric]["mean"].append(np.mean(data[config][metric]["mean"]) if data[config][metric]["mean"] else 0)
                nr_metric_data[metric]["std"].append(np.mean(data[config][metric]["std"]) if data[config][metric]["std"] else 0)

        # Plot no-reference metrics
        if any(nr_metric_data[metric]["mean"] for metric in nr_metrics):  # Only plot if NR metrics exist
            plot_nr_metrics(dataset, configs, nr_metric_data)