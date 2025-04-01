import os
import argparse
import datetime
from evaluation.eval import evaluate
from config.data_config import DATASETS, RESULTS_DIR

ALL_METRICS = [
    'PSNR', 'SSIM', 'MSSSIM', 'IWSSIM', 'VIF', 'FSIM', 'GMSD', 'MSGMSD', 'HAARPSI',
    'UQI', 'S3IM', 'BRISQUE'
]

def evaluate_dataset(dataset, dataset_info, metric_type="all", test_mode=False, timestamp=None):
    """
    Evaluates a dataset configuration-by-configuration and saves results progressively.

    :param dataset: Name of the dataset.
    :param dataset_info: Dictionary with dataset info from DATASETS.
    :param metric_type: "fr", "nr", or "all".
    :param test_mode: If True, uses test logic instead of real computation.
    :param timestamp: Consistent timestamp for per-dataset files.
    """
    results_dir = os.path.join(RESULTS_DIR, dataset)
    os.makedirs(results_dir, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    file_path = os.path.join(results_dir, f"{dataset}_results_{timestamp}.txt")

    with open(file_path, 'w') as f:
        if test_mode:
            f.write("TEST MODE - NO REAL METRICS\n")
        write_header(f)

    with open(file_path, 'a') as f:
        if dataset == "zenodo" or dataset in ["denoising_data", "pa_experiment_data"]:
            results = evaluate(dataset, None, None, None, metric_type, test_mode)
            for entry in results or []:
                write_result_entry(f, dataset, entry)

        else:
            if isinstance(dataset_info["path"], dict):
                for file_key in dataset_info["path"]:
                    for config, full_list in dataset_info["configs"].items():
                        for full_config in full_list:
                            results = evaluate(dataset, config, full_config, file_key, metric_type, test_mode)
                            for entry in results or []:
                                write_result_entry(f, dataset, entry)
            else:
                for config, full_list in dataset_info["configs"].items():
                    for full_config in full_list:
                        results = evaluate(dataset, config, full_config, None, metric_type, test_mode)
                        for entry in results or []:
                            write_result_entry(f, dataset, entry)

    print(f"Results saved to: {file_path}")

def write_header(f):
    header = "Dataset   Configuration   Ground Truth   Wavelength   " + \
             "   ".join([f"{m}_mean   {m}_std" for m in ALL_METRICS])
    f.write(header + "\n" + "-" * len(header) + "\n")

def write_result_entry(f, dataset, entry):
    if dataset == "MSFD":
        config, gt, wavelength, (metrics_mean, metrics_std) = entry
    elif dataset == "denoising_data":
        _, config, gt, wavelength, (metrics_mean, metrics_std) = entry
    elif dataset == "pa_experiment_data":
        _, config, gt, wavelength, (metrics_mean, metrics_std) = entry
    else:
        config, gt, (metrics_mean, metrics_std) = entry
        wavelength = "---"

    dataset_path = dataset
    line = f"{dataset_path:<30} {config:<15} {gt:<15} {wavelength:<11}"
    for metric in ALL_METRICS:
        mean = metrics_mean.get(metric, '---') if isinstance(metrics_mean, dict) else '---'
        std = metrics_std.get(metric, '---') if isinstance(metrics_std, dict) else '---'
        line += f"{mean:<10} {std:<8}"

    f.write(line + "\n")
    f.flush()
    os.fsync(f.fileno())

def main():
    parser = argparse.ArgumentParser(description="Evaluate datasets and configurations.")
    parser.add_argument("--datasets", nargs="+", help="Specify datasets to evaluate (e.g., SCD SWFD).")
    parser.add_argument("--metric_type", choices=["fr", "nr", "all"], default="all")
    parser.add_argument("--test", action="store_true", help="Run in test mode without real metric computation.")
    args = parser.parse_args()

    datasets = args.datasets if args.datasets else DATASETS.keys()
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for dataset in datasets:
        if dataset not in DATASETS:
            print(f"Unknown dataset '{dataset}'. Available: {list(DATASETS.keys())}")
            continue
        evaluate_dataset(dataset, DATASETS[dataset], metric_type=args.metric_type, test_mode=args.test, timestamp=timestamp)

if __name__ == "__main__":
    main()
