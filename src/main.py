import os
import argparse
from data_evaluation.eval import evaluate
from config.data_config import DATASETS


def evaluate_all_datasets(save_to_file=True, summary_path=None, selected_datasets=None):
    """
    Loop over selected datasets in DATASETS or all datasets if selected_datasets is None.
    Collect the metrics into a single summary.

    :param save_to_file: Whether to save the final summary to a text file.
    :param summary_path: Path to the final summary file (if save_to_file=True).
    :param selected_datasets: List of datasets to evaluate (default: all datasets).
    """
    all_results = []  # Store tuples for printing/writing
    datasets_to_evaluate = selected_datasets if selected_datasets else DATASETS.keys()

    for dataset in datasets_to_evaluate:
        dataset_info = DATASETS[dataset]
        # Handle datasets with multiple file keys
        if isinstance(dataset_info["path"], dict):
            for file_key in dataset_info["path"]:
                for geometry in dataset_info["configs"]:
                    partial_results = evaluate(dataset, geometry, file_key, save_results=False)
                    if partial_results:
                        for entry in partial_results:
                            all_results.append((dataset, file_key, geometry, entry))
        else:
            for geometry in dataset_info["configs"]:
                partial_results = evaluate(dataset, geometry, file_key=None, save_results=False)
                if partial_results:
                    for entry in partial_results:
                        all_results.append((dataset, None, geometry, entry))

    # Save results to file or print to console
    if save_to_file and summary_path:
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w") as f:
            f.write("Dataset   FileKey      Geometry   Mode     Wavelength   PSNR     SSIM\n")
            f.write("-----------------------------------------------------------------------\n")
            for (ds, fk, geom, entry) in all_results:
                if ds == "MSFD":
                    mode, wave, p, s = entry
                    wave_str = f"{wave}"
                else:
                    mode, p, s = entry
                    wave_str = "---"
                f.write(f"{ds:<9} {str(fk):<12} {geom:<9} {mode:<8} {wave_str:<11} {p:<7.3f} {s:<7.3f}\n")
        print(f"Summary results saved to {summary_path}")
    else:
        print("Dataset   FileKey      Geometry   Mode     Wavelength   PSNR     SSIM")
        print("-----------------------------------------------------------------------")
        for (ds, fk, geom, entry) in all_results:
            if ds == "MSFD":
                mode, wave, p, s = entry
                wave_str = str(wave)
            else:
                mode, p, s = entry
                wave_str = "---"
            print(f"{ds:<9} {str(fk):<12} {geom:<9} {mode:<8} {wave_str:<11} {p:<7.3f} {s:<7.3f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate datasets and configurations.")
    parser.add_argument("--datasets", nargs="+", help="Specify datasets to evaluate (e.g., mice phantom v_phantom).")
    parser.add_argument("--config", help="Configuration (e.g., lv128, ss64, sparse32).")
    parser.add_argument("--file_key", default=None, help="Optional file key for datasets like SWFD.")
    parser.add_argument("--summary", help="If provided, runs all datasets and saves results to this summary file.")
    parser.add_argument("--no_save", action="store_true", help="If provided, results are not saved to a file.")
    args = parser.parse_args()

    if args.summary:
        summary_path = os.path.join("results", "oadat", args.summary)
        selected_datasets = args.datasets if args.datasets else None
        evaluate_all_datasets(save_to_file=(not args.no_save), summary_path=summary_path, selected_datasets=selected_datasets)
    else:
        if not args.dataset or not args.config:
            print("You must provide --dataset and --config unless using --summary.")
            return

        if args.dataset not in DATASETS:
            print(f"Unknown dataset '{args.dataset}'. Available: {list(DATASETS.keys())}")
            return

        if isinstance(DATASETS[args.dataset]["path"], dict):
            if not args.file_key or args.file_key not in DATASETS[args.dataset]["path"]:
                print(f"The dataset '{args.dataset}' requires a valid --file_key.")
                return

        if args.config not in DATASETS[args.dataset]["configs"]:
            print(f"Invalid configuration '{args.config}' for dataset '{args.dataset}'.")
            return

        results = evaluate(args.dataset, args.config, args.file_key, save_results=not args.no_save)
        if not results:
            print("No results returned. Ensure correct dataset, configuration, and file_key.")


if __name__ == "__main__":
    main()