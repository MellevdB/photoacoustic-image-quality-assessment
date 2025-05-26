import os
import argparse
import datetime
import pandas as pd
from evaluation.eval import evaluate
from config.data_config import DATASETS, RESULTS_DIR
import numpy as np
from PIL import Image

ALL_METRICS = [
    'PSNR', 'SSIM', 'MSSSIM', 'IWSSIM', 'VIF', 'FSIM', 'GMSD', 'MSGMSD', 'HAARPSI',
    'UQI', 'S3IM', "TV", "BRISQUE", "CLIP-IQA"
]
NORMALIZED_METRICS = ["PSNR_norm", "GMSD_norm", "MSGMSD_norm", "BRISQUE_norm"]

def evaluate_dataset(dataset, dataset_info, metric_type="all", test_mode=False, timestamp=None):
    results_dir = os.path.join(RESULTS_DIR, dataset)
    os.makedirs(results_dir, exist_ok=True)

    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    file_path = os.path.join(results_dir, f"{dataset}_results_{timestamp}.txt")

    with open(file_path, 'w') as f:
        if test_mode:
            f.write("TEST MODE - NO REAL METRICS\n")
        write_header(f)

    all_results = []

    with open(file_path, 'a') as f:
        if dataset == "zenodo" or dataset in ["denoising_data", "pa_experiment_data"]:
            results = evaluate(dataset, None, None, None, metric_type, test_mode)
            all_results.extend(results)
            for entry in results or []:
                write_result_entry(f, dataset, entry)
        else:
            if isinstance(dataset_info["path"], dict):
                for file_key in dataset_info["path"]:
                    for config, full_list in dataset_info["configs"].items():
                        for full_config in full_list:
                            results = evaluate(dataset, config, full_config, file_key, metric_type, test_mode)
                            all_results.extend(results)
                            for entry in results or []:
                                write_result_entry(f, dataset, entry)
            else:
                for config, full_list in dataset_info["configs"].items():
                    for full_config in full_list:
                        results = evaluate(dataset, config, full_config, None, metric_type, test_mode)
                        all_results.extend(results)
                        for entry in results or []:
                            write_result_entry(f, dataset, entry)

    print(f"Results saved to: {file_path}")

    # Save per-image metrics to CSV
    per_image_rows = []

    image_dir = os.path.join(results_dir, "images_used")
    os.makedirs(image_dir, exist_ok=True)

    for entry in all_results:
        if dataset == "MSFD":
            config, gt, wavelength, (metrics_mean, metrics_std, raw_metrics, image_ids) = entry
        elif dataset in ["denoising_data", "pa_experiment_data"]:
            path, config, gt, wavelength, (metrics_mean, metrics_std, raw_metrics, image_ids) = entry
        else:
            config, gt, (metrics_mean, metrics_std, raw_metrics, image_ids) = entry
            wavelength = "---"

        for idx, image_id in enumerate(image_ids):
            # Decide how to construct image_path
            if dataset in ["denoising_data", "pa_experiment_data", "zenodo"] and os.path.isfile(image_id):
                image_path = image_id  # keep original path
            else:
                if "RECON_IMAGE" in raw_metrics:
                    image_array = raw_metrics["RECON_IMAGE"][idx]
                    base_name = os.path.basename(image_id).replace('.png', '')  # Strip path and .png extension if present
                    image_name = f"{base_name}.png"
                    image_path = os.path.join(image_dir, image_name)
                    Image.fromarray((np.clip(image_array, 0, 1) * 255).astype(np.uint8)).save(image_path)
                else:
                    image_path = image_id  # fallback

            row = {
                "dataset": dataset,
                "configuration": config,
                "ground_truth": gt,
                "wavelength": wavelength,
                "image_path": image_path
            }

            for metric in ALL_METRICS:
                if raw_metrics.get(metric) is not None:
                    row[metric] = raw_metrics[metric][idx]
                else:
                    row[metric] = float('nan')

            for metric in NORMALIZED_METRICS:
                if raw_metrics.get(metric) is not None:
                    row[metric] = raw_metrics[metric][idx]

            per_image_rows.append(row)

    if per_image_rows:
        df = pd.DataFrame(per_image_rows)
        csv_path = os.path.join(results_dir, f"{dataset}_per_image_metrics_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        print(f"Per-image metric scores saved to: {csv_path}")

def write_header(f):
    header = "Dataset   Configuration   Ground Truth   Wavelength   " + \
             "   ".join([f"{m}_mean   {m}_std" for m in ALL_METRICS])
    f.write(header + "\n" + "-" * len(header) + "\n")

def write_result_entry(f, dataset, entry):
    if dataset == "MSFD":
        config, gt, wavelength, (metrics_mean, metrics_std, *_rest) = entry
    elif dataset in ["denoising_data", "pa_experiment_data"]:
        path, config, gt, wavelength, (metrics_mean, metrics_std, *_rest) = entry
    else:
        config, gt, (metrics_mean, metrics_std, *_rest) = entry
        wavelength = "---"
    
    if dataset == "pa_experiment_data":
        line = f"{path:<30} {config:<15} {gt:<15} {wavelength:<11}"
    else:
        line = f"{dataset:<30} {config:<15} {gt:<15} {wavelength:<11}"
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