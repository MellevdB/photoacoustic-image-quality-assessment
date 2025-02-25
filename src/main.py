import os
import argparse
import datetime
from data_evaluation.eval import evaluate
from config.data_config import DATASETS, RESULTS_DIR

def evaluate_all_datasets(selected_datasets=None, metric_type="all", fake_results=False):
    """
    Evaluates datasets and progressively saves results.

    :param selected_datasets: List of datasets to evaluate (default: all datasets).
    :param metric_type: "fr" for full-reference, "nr" for no-reference, or "all".
    :param fake_results: If True, use fake metrics instead of real calculations.
    """
    datasets_to_evaluate = selected_datasets if selected_datasets else DATASETS.keys()

    # Create a single timestamp for consistency
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # If all datasets are being evaluated together, use a single file
    if selected_datasets is None:
        results_dir = os.path.join(RESULTS_DIR, "all_datasets")
        os.makedirs(results_dir, exist_ok=True)
        file_path = os.path.join(results_dir, f"all_datasets_results_{timestamp}.txt")

        with open(file_path, 'w') as f:
            if fake_results:
                f.write("FAKE RESULTS USED\n")
            write_header(f, metric_type)

        for dataset in datasets_to_evaluate:
            dataset_info = DATASETS[dataset]
            evaluate_dataset(dataset, dataset_info, metric_type, fake_results, file_path=file_path)

        print(f"All results saved to: {file_path}")

    else:
        # Evaluate each dataset separately, saving results in their own folders
        for dataset in datasets_to_evaluate:
            dataset_info = DATASETS[dataset]
            evaluate_dataset(dataset, dataset_info, metric_type, fake_results, timestamp=timestamp)


def evaluate_dataset(dataset, dataset_info, metric_type="all", fake_results=False, file_path=None, timestamp=None):
    """
    Evaluates a dataset configuration-by-configuration and saves results progressively.

    :param dataset: Name of the dataset.
    :param dataset_info: Dictionary with dataset info from DATASETS.
    :param metric_type: "fr", "nr", or "all".
    :param fake_results: If True, use fake metrics instead of real calculations.
    :param file_path: If specified, all results will be saved in this single file (used for all_datasets case).
    :param timestamp: Consistent timestamp for per-dataset files.
    """
    # If file_path is None, we are saving per dataset
    if file_path is None:
        results_dir = os.path.join(RESULTS_DIR, dataset)
        os.makedirs(results_dir, exist_ok=True)

        if timestamp is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        file_path = os.path.join(results_dir, f"{dataset}_results_{timestamp}.txt")

        with open(file_path, 'w') as f:
            if fake_results:
                f.write("FAKE RESULTS USED\n")
            write_header(f, metric_type)

    # Open file in append mode for progressive saving
    with open(file_path, 'a') as f:
        if dataset not in ["denoising_data", "pa_experiment_data"]:
            if isinstance(dataset_info["path"], dict):
                for file_key in dataset_info["path"]:
                    for config, config_values in dataset_info["configs"].items():
                        for full_config in config_values:
                            partial_results = evaluate(dataset, config, full_config, file_key, metric_type=metric_type, fake_results=fake_results)
                            if partial_results:
                                for entry in partial_results:
                                    write_result_entry(f, dataset, entry, metric_type)
            else:
                for config, config_values in dataset_info["configs"].items():
                    for full_config in config_values:
                        partial_results = evaluate(dataset, config, full_config, file_key=None, metric_type=metric_type, fake_results=fake_results)
                        if partial_results:
                            for entry in partial_results:
                                write_result_entry(f, dataset, entry, metric_type)

        else:
            # New datasets: Process them using modified logic
            results = evaluate(dataset, None, None, None, metric_type, fake_results)
            if results:
                for entry in results:
                    write_result_entry(f, dataset, entry, metric_type)

        print(f"Results saved progressively to: {file_path}")


def write_header(f, metric_type):
    """
    Writes the header for the results file.

    :param f: Open file object
    :param metric_type: "fr", "nr", or "all"
    """
    metric_headers = ['FSIM', 'UQI', 'PSNR', 'SSIM', 'VIF', 'S3IM'] if metric_type == "fr" else \
                     ['BRISQUE'] if metric_type == "nr" else \
                     ['FSIM', 'UQI', 'PSNR', 'SSIM', 'VIF', 'S3IM', 'BRISQUE']
    header = "Dataset   Configuration   Ground Truth   Wavelength   " + "   ".join([f"{m}_mean   {m}_std" for m in metric_headers])
    f.write(header + "\n" + "-" * len(header) + "\n")


def write_result_entry(f, dataset, entry, metric_type):
    """
    Writes a single result entry to an open file and immediately flushes.

    :param f: Open file object
    :param dataset: Dataset name
    :param entry: Tuple containing configuration details and metric results
    :param metric_type: "fr", "nr", or "all"
    """
    metric_headers = ['FSIM', 'UQI', 'PSNR', 'SSIM', 'VIF', 'S3IM'] if metric_type == "fr" else \
                     ['BRISQUE'] if metric_type == "nr" else \
                     ['FSIM', 'UQI', 'PSNR', 'SSIM', 'VIF', 'S3IM', 'BRISQUE']

    # Preserve old behavior for existing datasets
    if dataset not in ["denoising_data", "pa_experiment_data"]:
        if dataset == "MSFD":
            config, ground_truth, wavelength, (metrics_mean, metrics_std) = entry
        else:
            config, ground_truth, (metrics_mean, metrics_std) = entry
            wavelength = "---"
        dataset_path = dataset

    # New datasets: format entries differently
    elif dataset == "denoising_data":
        
        _, config, ground_truth, wavelength, (metrics_mean, metrics_std) = entry
        dataset_path = "denoising_data/nne/train"


    elif dataset == "pa_experiment_data":
        dataset_path, config, ground_truth, wavelength, (metrics_mean, metrics_std) = entry

    # Construct line for saving results
    line = f"{dataset_path:<30} {config:<10} {ground_truth:<15} {wavelength:<11}"
    for metric in metric_headers:
        mean = metrics_mean.get(metric, float('nan')) if isinstance(metrics_mean, dict) else float('nan')
        std = metrics_std.get(metric, float('nan')) if isinstance(metrics_std, dict) else float('nan')
        line += f"{mean:<10.3f} {std:<8.3f}"

    f.write(line + "\n")
    f.flush()  # Force immediate write to disk
    os.fsync(f.fileno())  # Ensure OS writes data immediately


def main():
    parser = argparse.ArgumentParser(description="Evaluate datasets and configurations.")
    parser.add_argument("--datasets", nargs="+", help="Specify datasets to evaluate (e.g., SCD, SWFD).")
    parser.add_argument("--metric_type", choices=["fr", "nr", "all"], default="all", help="Choose 'fr' for full-reference, 'nr' for no-reference, or 'all'.")
    parser.add_argument("--fake_results", action="store_true", help="Use fake metric values for fast testing.")
    args = parser.parse_args()

    selected_datasets = args.datasets if args.datasets else None

    if not selected_datasets:
        print("No datasets specified. Evaluating all datasets...")
    else:
        for dataset in selected_datasets:
            if dataset not in DATASETS:
                print(f"Unknown dataset '{dataset}'. Available: {list(DATASETS.keys())}")
                return

    evaluate_all_datasets(selected_datasets=selected_datasets, metric_type=args.metric_type, fake_results=args.fake_results)

if __name__ == "__main__":
    main()