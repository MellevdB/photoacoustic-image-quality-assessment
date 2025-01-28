import os
import argparse
from data_evaluation.eval import evaluate
from config.data_config import DATASETS, RESULTS_DIR
import datetime


def evaluate_all_datasets(save_to_file=True, selected_datasets=None):
    """
    Loop over selected datasets in DATASETS or all datasets if selected_datasets is None.
    Collect the metrics into a single summary.

    :param save_to_file: Whether to save the final summary to a file.
    :param selected_datasets: List of datasets to evaluate (default: all datasets).
    """
    datasets_to_evaluate = selected_datasets if selected_datasets else DATASETS.keys()
    all_results = []

    for dataset in datasets_to_evaluate:
        dataset_info = DATASETS[dataset]
        results = []

        # Handle datasets with multiple file keys (like SWFD)
        if isinstance(dataset_info["path"], dict):
            for file_key in dataset_info["path"]:
                for config, config_values in dataset_info["configs"].items():
                    for full_config in config_values:
                        partial_results = evaluate(dataset, config, full_config, file_key, save_results=False)
                        if partial_results:
                            # Directly extend results with partial_results
                            results.extend(partial_results)
        else:
            for config, config_values in dataset_info["configs"].items():
                for full_config in config_values:
                    partial_results = evaluate(dataset, config, full_config, file_key=None, save_results=False)
                    if partial_results:
                        # Directly extend results with partial_results
                        results.extend(partial_results)
                        print("results", results)

        # Save results to file
        if save_to_file and results:
            if selected_datasets and len(selected_datasets) == 1:
                # Save to dataset-specific results file
                results_path = os.path.join(RESULTS_DIR, dataset, f"{dataset}_results.txt")
                save_results_to_file(results, results_path, dataset)
            else:  # Collect results for all datasets
                all_results.extend([(dataset, *entry) for entry in results])
        
    # Save all results in a single file when running all datasets
    if save_to_file and not selected_datasets and all_results:
        all_results_path = os.path.join(RESULTS_DIR, "all_datasets", "all_datasets_results.txt")
        save_results_to_file(all_results, all_results_path, "all_datasets")


def save_results_to_file(results, results_path, dataset_name):
    # Generate timestamp for unique filenames
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base, ext = os.path.splitext(results_path)
    results_path = f"{base}_{timestamp}{ext}"

    os.makedirs(os.path.dirname(results_path), exist_ok=True)

    # Extract dynamic headers from metrics
    example_metrics = results[0][3] if dataset_name == "MSFD" else results[0][2]
    metric_headers = list(example_metrics.keys())
    headers = ["Dataset", "Configuration", "Ground Truth"]
    if dataset_name == "MSFD":
        headers.append("Wavelength")
    headers.extend(metric_headers)

    # Write headers
    header_line = "   ".join(headers)
    with open(results_path, "w") as f:
        f.write(header_line + "\n")
        f.write("-" * len(header_line) + "\n")

        # Write results
        for entry in results:
            if dataset_name == "MSFD":
                config, ground_truth, wavelength, metrics = entry
                row = f"{dataset_name:<9} {config:<20} {ground_truth:<15} {wavelength:<11} "
            else:
                config, ground_truth, metrics = entry
                row = f"{dataset_name:<9} {config:<20} {ground_truth:<15} "

            metrics_str = " ".join([f"{float(value):<7.3f}" if isinstance(value, (int, float)) else "---" for value in metrics.values()])
            row += metrics_str
            f.write(row + "\n")

    print(f"Results saved to {results_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate datasets and configurations.")
    parser.add_argument("--datasets", nargs="+", help="Specify datasets to evaluate (e.g., SCD, SWFD).")
    parser.add_argument("--no_save", action="store_true", help="If provided, results are not saved to a file.")
    args = parser.parse_args()

    selected_datasets = args.datasets if args.datasets else None

    if not selected_datasets:
        print("No datasets specified. Evaluating all datasets...")
    else:
        for dataset in selected_datasets:
            if dataset not in DATASETS:
                print(f"Unknown dataset '{dataset}'. Available: {list(DATASETS.keys())}")
                return

    evaluate_all_datasets(save_to_file=(not args.no_save), selected_datasets=selected_datasets)


if __name__ == "__main__":
    main()