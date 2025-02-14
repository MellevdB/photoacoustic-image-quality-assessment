import os
import argparse
import datetime
from data_evaluation.eval import evaluate
from config.data_config import DATASETS, RESULTS_DIR

def evaluate_all_datasets(selected_datasets=None, metric_type="all", fake_results=False):
    """
    Loop over selected datasets in DATASETS or all datasets if selected_datasets is None.
    Collect the metrics into a single summary and save the results.

    :param selected_datasets: List of datasets to evaluate (default: all datasets).
    :param metric_type: "fr" for full-reference metrics, "nr" for no-reference metrics, or "all".
    :param fake_results: If True, use fake metrics instead of real calculations.
    """
    datasets_to_evaluate = selected_datasets if selected_datasets else DATASETS.keys()
    all_results = []

    for dataset in datasets_to_evaluate:
        dataset_info = DATASETS[dataset]
        results = evaluate_dataset(dataset, dataset_info, metric_type=metric_type, fake_results=fake_results)

        if results:
            if selected_datasets:
                #  Save only per dataset when running a single dataset
                results_path = os.path.join(RESULTS_DIR, dataset, f"{dataset}_results.txt")
                save_results_to_file(results, results_path, dataset, metric_type, fake_results)
            else:
                #  Accumulate results for all datasets
                all_results.extend([(dataset, *entry) for entry in results])

    # Only save to `all_datasets` when evaluating ALL datasets
    if not selected_datasets and all_results:
        all_results_path = os.path.join(RESULTS_DIR, "all_datasets", "all_datasets_results.txt")
        save_results_to_file(all_results, all_results_path, "all_datasets", metric_type, fake_results)

def evaluate_dataset(dataset, dataset_info, metric_type="all", fake_results=False):
    """
    Helper function to evaluate a specific dataset by iterating through file keys/config values.
    Returns a list of results for that dataset.

    :param dataset: Name of the dataset (string).
    :param dataset_info: Dictionary with dataset info from DATASETS.
    :param metric_type: "fr", "nr", or "all".
    :param fake_results: If True, use fake metrics instead of real calculations.
    :return: List of results for this dataset.
    """
    results = []
    
    if isinstance(dataset_info["path"], dict):
        for file_key in dataset_info["path"]:
            for config, config_values in dataset_info["configs"].items():
                for full_config in config_values:
                    partial_results = evaluate(dataset, config, full_config, file_key, metric_type=metric_type, fake_results=fake_results)
                    if partial_results:
                        results.extend(partial_results)
    else:
        for config, config_values in dataset_info["configs"].items():
            for full_config in config_values:
                partial_results = evaluate(dataset, config, full_config, file_key=None, metric_type=metric_type, fake_results=fake_results)
                if partial_results:
                    results.extend(partial_results)
    return results

def save_results_to_file(results, file_path, dataset_name, metric_type="all", fake_results=False):
    """
    Save evaluation results to a text file with a timestamp in the filename.

    Args:
        results: List of tuples containing:
                 - For MSFD: (dataset, config, ground_truth, wavelength, metrics_mean, metrics_std)
                 - For other datasets: (dataset, config, ground_truth, metrics_mean, metrics_std)
        file_path: Path to save the results file.
        dataset_name: Name of the dataset being evaluated or "all_datasets".
        metric_type: "fr", "nr", or "all".
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_path = os.path.dirname(file_path)
    base_name = os.path.basename(file_path).replace(".txt", "")
    new_file_path = os.path.join(dir_path, f"{base_name}_{timestamp}.txt")

    # Define metric headers based on metric_type
    if metric_type == "fr":
        metric_headers = ['FSIM', 'UQI', 'PSNR', 'SSIM', 'VIF', 'S3IM']
    elif metric_type == "nr":
        metric_headers = ['BRISQUE']  # Expand with 'NIQE', 'NIQE-K' if needed
    elif metric_type == "all":
        metric_headers = ['FSIM', 'UQI', 'PSNR', 'SSIM', 'VIF', 'S3IM', 'BRISQUE']  # Expand if needed
    else:
        metric_headers = []

    header = "Dataset   Configuration   Ground Truth   Wavelength   " + "   ".join([f"{m}_mean   {m}_std" for m in metric_headers])

    result_lines = []
    for entry in results:
        if dataset_name == "all_datasets":
            dataset, *entry_data = entry  
        else:
            dataset = dataset_name
            entry_data = entry

        if dataset == "MSFD":
            config, ground_truth, wavelength, (metrics_mean, metrics_std) = entry_data
        else:
            config, ground_truth, (metrics_mean, metrics_std) = entry_data
            wavelength = "---"

        line = f"{dataset:<9} {config:<20} {ground_truth:<15} {wavelength:<11}"
        for metric in metric_headers:
            mean = metrics_mean.get(metric, float('nan')) if isinstance(metrics_mean, dict) else float('nan')
            std = metrics_std.get(metric, float('nan')) if isinstance(metrics_std, dict) else float('nan')
            line += f"{mean:<10.3f} {std:<8.3f}"
        result_lines.append(line)

    os.makedirs(dir_path, exist_ok=True)
    with open(new_file_path, 'w') as f:
        if fake_results:
            f.write("Fake results used. \n")
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        f.write("\n".join(result_lines))

    print(f"Results saved to: {new_file_path}")

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