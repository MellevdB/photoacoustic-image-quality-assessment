import os
import argparse
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

    for dataset in datasets_to_evaluate:
        dataset_info = DATASETS[dataset]
        results = []

        # Handle datasets with multiple file keys (like SWFD)
        if isinstance(dataset_info["path"], dict):
            for file_key in dataset_info["path"]:
                for config, config_values in dataset_info["configs"].items():
                    for full_config in config_values:
                        partial_results = evaluate(dataset, full_config, file_key, save_results=False)
                        if partial_results:
                            results.extend([(full_config, entry) for entry in partial_results])
        else:
            for config, config_values in dataset_info["configs"].items():
                for full_config in config_values:
                    partial_results = evaluate(dataset, full_config, file_key=None, save_results=False)
                    if partial_results:
                        results.extend([(full_config, entry) for entry in partial_results])

        # Save results to file
        if save_to_file and results:
            if selected_datasets and len(selected_datasets) == 1:
                # Save to dataset-specific results file
                results_path = os.path.join(RESULTS_DIR, dataset, f"{dataset}_results.txt")
            else:
                # Save to all_datasets directory
                results_path = os.path.join(RESULTS_DIR, "all_datasets", f"{dataset}_results.txt")

            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            with open(results_path, "w") as f:
                f.write("Dataset   Configuration       Wavelength   PSNR     SSIM\n")
                f.write("------------------------------------------------------\n")
                for config, entry in results:
                    if dataset == "MSFD":
                        mode, wave, psnr, ssim = entry
                        wave_str = f"{wave}"
                    else:
                        mode, psnr, ssim = entry
                        wave_str = "---"
                    f.write(f"{dataset:<9} {config:<20} {wave_str:<11} {psnr:<7.3f} {ssim:<7.3f}\n")
            print(f"Results saved to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate datasets and configurations.")
    parser.add_argument("--datasets", nargs="+", help="Specify datasets to evaluate (e.g., SCD, SWFD).")
    parser.add_argument("--config", help="Configuration (e.g., lv128, ss64, sparse32).")
    parser.add_argument("--file_key", default=None, help="Optional file key for datasets like SWFD.")
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