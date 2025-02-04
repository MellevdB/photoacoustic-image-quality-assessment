import os
import argparse
import datetime
from data_evaluation.eval import evaluate
from config.data_config import DATASETS, RESULTS_DIR

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
        results = evaluate_dataset(dataset, dataset_info)

        # If specifically evaluating only one dataset, save its results
        if save_to_file and results and selected_datasets and len(selected_datasets) == 1:
            results_path = os.path.join(RESULTS_DIR, dataset, f"{dataset}_results.txt")
            save_results_to_file(results, results_path, dataset)
        else:
            # Accumulate in all_results if we're evaluating multiple or all datasets
            # For multi-dataset results, we prepend the dataset name to each entry
            all_results.extend([(dataset, *entry) for entry in results])

    # Save all results in a single file when running all datasets
    if save_to_file and not selected_datasets and all_results:
        all_results_path = os.path.join(RESULTS_DIR, "all_datasets", "all_datasets_results.txt")
        save_results_to_file(all_results, all_results_path, "all_datasets")

def evaluate_dataset(dataset, dataset_info):
    """
    Helper function to evaluate a specific dataset by iterating through file keys/config values.
    Returns a list of results for that dataset.

    :param dataset: Name of the dataset (string).
    :param dataset_info: Dictionary with dataset info from DATASETS.
    :return: List of results for this dataset. Each result entry is:
        - For MSFD: (full_config, ground_truth_key, wavelength, metrics)
        - Otherwise: (full_config, ground_truth_key, metrics)
    """
    results = []
    # Handle datasets with multiple file keys (like SWFD with dict paths)
    if isinstance(dataset_info["path"], dict):
        for file_key in dataset_info["path"]:
            for config, config_values in dataset_info["configs"].items():
                for full_config in config_values:
                    partial_results = evaluate(dataset, config, full_config, file_key, save_results=False)
                    if partial_results:
                        results.extend(partial_results)
    else:
        # Single path
        for config, config_values in dataset_info["configs"].items():
            for full_config in config_values:
                partial_results = evaluate(dataset, config, full_config, file_key=None, save_results=False)
                if partial_results:
                    results.extend(partial_results)
                    print("results", results)

    return results

def save_results_to_file(results, file_path, dataset_name):
    """
    Save evaluation results to a text file with timestamp in the filename.
    
    Args:
        results: List of tuples containing (config, ground_truth, (metrics_mean, metrics_std))
        file_path: Path to save the results file
        dataset_name: Name of the dataset being evaluated
    """
    # Add timestamp to filename
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dir_path = os.path.dirname(file_path)
    base_name = os.path.basename(file_path).replace(".txt", "")
    new_file_path = os.path.join(dir_path, f"{base_name}_{timestamp}.txt")
    
    # Define the header
    header = "Dataset   Configuration   Ground Truth   " + \
             "FSIM_mean   FSIM_std   NQM_mean   NQM_std   " + \
             "PSNR_mean   PSNR_std   SSIM_mean   SSIM_std   " + \
             "VIF_mean   VIF_std   S3IM_mean   S3IM_std"
    
    # Prepare the results lines
    result_lines = []
    for config, ground_truth, (metrics_mean, metrics_std) in results:
        # Create the base line with dataset, config and ground truth
        line = f"{dataset_name:<9} {config:<20} {ground_truth:<15}"
        
        # Add each metric with mean and std
        for metric in ['FSIM', 'NQM', 'PSNR', 'SSIM', 'VIF', 'S3IM']:
            mean = metrics_mean.get(metric, float('nan'))
            std = metrics_std.get(metric, float('nan'))
            line += f"{mean:<10.3f} {std:<8.3f}"
        
        result_lines.append(line)
    
    # Write to file
    with open(new_file_path, 'w') as f:
        # Write header
        f.write(header + "\n")
        f.write("-" * len(header) + "\n")
        
        # Write results
        f.write("\n".join(result_lines))
    
    print(f"Results saved to: {new_file_path}")

def format_result_line(entry, dataset_name, metric_headers):
    """
    Format a single result entry into a line of text, including the possibility
    of a wavelength column for MSFD or 'all_datasets'.

    :param entry: A tuple containing result info.
    :param dataset_name: e.g. 'SCD', 'MSFD', or 'all_datasets'.
    :param metric_headers: List of metric keys to display in the row.
    :return: A formatted string that can be written to the file.
    """
    # For "all_datasets", we can have either 4 or 5 items in the tuple:
    if dataset_name == "all_datasets":
        if len(entry) == 4:
            # (dataset, config, ground_truth, metrics)
            dataset, config, ground_truth, metrics = entry
            # We might still want to handle the wavelength column if it exists among other rows
            # so we print `---` for the wavelength
            row_str = f"{dataset:<9} {config:<20} {ground_truth:<15} {'---':<11}"
        else:
            # (dataset, config, ground_truth, wavelength, metrics)
            dataset, config, ground_truth, wavelength, metrics = entry
            row_str = f"{dataset:<9} {config:<20} {ground_truth:<15} {wavelength:<11}"

    else:
        # Single dataset scenario
        if dataset_name == "MSFD":
            # (config, ground_truth, wavelength, metrics)
            config, ground_truth, wavelength, metrics = entry
            row_str = f"{dataset_name:<9} {config:<20} {ground_truth:<15} {wavelength:<11}"
        else:
            # (config, ground_truth, metrics)
            config, ground_truth, metrics = entry
            row_str = f"{dataset_name:<9} {config:<20} {ground_truth:<15}"

    # Append metrics
    metrics_str = " ".join(
        f"{float(value):<7.3f}" if isinstance(value, (int, float)) else "---"
        for key, value in metrics.items()
        if key in metric_headers
    )

    return row_str + " " + metrics_str

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