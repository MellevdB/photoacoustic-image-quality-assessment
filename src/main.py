# main.py

import os
import argparse

from evaluation_oadat.eval_translation import evaluate
from config.data_config import DATASETS


def evaluate_all_datasets(save_to_file=True, summary_path=None):
    """
    Loop over ALL datasets in DATASETS (SCD, SWFD, MSFD),
    all file_keys (if any), and all geometry config entries.
    Collect the metrics into a single summary.

    :param save_to_file: Whether to save the final summary to a text file.
    :param summary_path: Path to the final summary file (if save_to_file=True).
    """
    all_results = []  # We'll store tuples for printing/writing

    for dataset in DATASETS:
        dataset_info = DATASETS[dataset]
        # Might be a single path or a dict of multiple paths
        if isinstance(dataset_info["path"], dict):
            # e.g., SWFD
            for file_key in dataset_info["path"]:
                for geometry in dataset_info["configs"]:
                    # Evaluate once for each geometry
                    partial_results = evaluate(dataset, geometry, file_key, save_results=False)
                    # partial_results might be (mode, psnr, ssim) or (mode, wavelength, psnr, ssim)
                    if partial_results:
                        for entry in partial_results:
                            all_results.append((dataset, file_key, geometry, entry))
        else:
            # e.g., SCD or MSFD
            for geometry in dataset_info["configs"]:
                partial_results = evaluate(dataset, geometry, file_key=None, save_results=False)
                if partial_results:
                    for entry in partial_results:
                        all_results.append((dataset, None, geometry, entry))

    # Now optionally write to summary file
    if save_to_file and summary_path:
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w") as f:
            # Write a simple header
            f.write("Dataset   FileKey      Geometry   Mode     Wavelength   PSNR     SSIM\n")
            f.write("-----------------------------------------------------------------------\n")
            for (ds, fk, geom, entry) in all_results:
                if ds == "MSFD":
                    # entry is (mode, wavelength, psnr, ssim)
                    mode, wave, p, s = entry
                    wave_str = f"{wave}"
                else:
                    # entry is (mode, psnr, ssim)
                    mode, p, s = entry
                    wave_str = "---"

                f.write(f"{ds:<9} {str(fk):<12} {geom:<9} {mode:<8} {wave_str:<11} {p:<7.3f} {s:<7.3f}\n")
        print(f"Summary results saved to {summary_path}")

    else:
        # Just print to stdout
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
    parser = argparse.ArgumentParser(description="Evaluate OADAT datasets (SCD, SWFD, MSFD).")
    parser.add_argument("--dataset", help="Dataset name (SCD, SWFD, MSFD).")
    parser.add_argument("--geometry", help="Geometry/config (e.g., lv128, ss64, etc.).")
    parser.add_argument("--file_key", default=None,
                        help="For SWFD, one of 'multisegment' or 'semicircle'. Not needed for SCD/MSFD.")
    parser.add_argument("--summary", help="If provided, runs all datasets and saves results to this summary file.")
    parser.add_argument("--no_save", action="store_true",
                        help="If given, do not save individual results to a text file.")
    args = parser.parse_args()

    if args.summary:
        # Evaluate everything and write to summary
        summary_path = os.path.join("results", "oadat", args.summary)
        evaluate_all_datasets(save_to_file=(not args.no_save), summary_path=summary_path)
    else:
        # We do a single dataset + geometry + optional file_key
        if not args.dataset or not args.geometry:
            print("You must provide both --dataset and --geometry (unless using --summary).")
            return

        if args.dataset not in DATASETS:
            print(f"Unknown dataset '{args.dataset}'. Available: {list(DATASETS.keys())}")
            return

        # If dataset has multiple file keys (like SWFD), we must validate
        dataset_info = DATASETS[args.dataset]
        if isinstance(dataset_info["path"], dict):
            valid_keys = list(dataset_info["path"].keys())
            if not args.file_key or args.file_key not in valid_keys:
                print(f"The dataset '{args.dataset}' requires --file_key. Options: {valid_keys}")
                return

        # Validate geometry
        if args.geometry not in dataset_info["configs"]:
            print(f"Invalid geometry '{args.geometry}' for dataset '{args.dataset}'!")
            print(f"Available: {list(dataset_info['configs'].keys())}")
            return

        # Run the evaluation
        results = evaluate(args.dataset, args.geometry, file_key=args.file_key,
                           save_results=(not args.no_save))
        if not results:
            print("No results returned (possibly missing HDF5 keys).")

if __name__ == "__main__":
    main()