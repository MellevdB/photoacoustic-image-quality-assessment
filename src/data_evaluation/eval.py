import os
import scipy.io as sio
import h5py
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from preprocessing_data.filterBandPass import sigMatFilter
from preprocessing_data.normalize import sigMatNormalize
from config.data_config import DATASETS, DATA_DIR, RESULTS_DIR


def load_mat_file(file_path, key):
    try:
        data = sio.loadmat(file_path)
        return data[key]
    except NotImplementedError:
        with h5py.File(file_path, "r") as f:
            return f[key][()]


def calculate_metrics(y_pred, y_true):
    data_range = y_true.max() - y_true.min()
    psnr = peak_signal_noise_ratio(y_true, y_pred, data_range=data_range)
    ssim = structural_similarity(y_true, y_pred, data_range=data_range)
    return psnr, ssim


def evaluate(dataset, full_config, file_key=None, save_results=True):
    """
    Evaluate a specific dataset and configuration for PSNR and SSIM.

    :param dataset: Dataset name (e.g., SCD, SWFD, MSFD, mice, phantom, v_phantom).
    :param full_config: Full configuration key (e.g., vc,lv128_BP).
    :param file_key: Optional key for datasets like SWFD (e.g., "multisegment" or "semicircle").
    :param save_results: Whether to save the results to a file.
    :return: List of tuples containing (configuration, psnr, ssim) or (configuration, wavelength, psnr, ssim) for MSFD.
    """
    results = []
    dataset_info = DATASETS[dataset]
    print(f"Processing dataset={dataset}, config={full_config}, file_key={file_key}")

    # Handle HDF5 datasets (e.g., SCD, SWFD, MSFD)
    if dataset in ["SCD", "SWFD", "MSFD"]:
        if isinstance(dataset_info["path"], dict):
            # For datasets with multiple file keys
            data_path = dataset_info["path"].get(file_key)
            if not data_path:
                print(f"[WARNING] File key '{file_key}' not found for dataset '{dataset}'. Skipping...")
                return results
        else:
            # For datasets with a single path
            data_path = dataset_info["path"]

        if not os.path.isfile(data_path):
            print(f"[WARNING] File not found: {data_path}. Skipping...")
            return results

        with h5py.File(data_path, "r") as data:
            if dataset == "MSFD":
                # MSFD: Iterate over wavelengths
                for wavelength, ground_truth_key in dataset_info["ground_truth"]["wavelengths"].items():
                    if full_config in data and ground_truth_key in data:
                        print(f"Processing wavelength={wavelength}")
                        y_pred = sigMatNormalize(sigMatFilter(data[full_config][:]))
                        y_true = sigMatNormalize(sigMatFilter(data[ground_truth_key][:]))
                        psnr, ssim = calculate_metrics(y_pred, y_true)
                        results.append((full_config, wavelength, psnr, ssim))
            else:
                # SCD/SWFD: Match ground truth based on file_key or configuration
                ground_truth_key = None
                if file_key:
                    # Use the ground_truth mapping for SWFD
                    ground_truth_key = dataset_info["ground_truth"].get(file_key)
                else:
                    # Dynamically resolve ground_truth for SCD
                    for key in dataset_info["ground_truth"]:
                        if key in full_config:
                            ground_truth_key = dataset_info["ground_truth"][key]
                            break

                if not ground_truth_key:
                    print(f"[ERROR] Ground truth key not found for config '{full_config}' and file_key '{file_key}'. Skipping...")
                    return results

                if full_config in data and ground_truth_key in data:
                    y_pred = sigMatNormalize(sigMatFilter(data[full_config][:]))
                    y_true = sigMatNormalize(sigMatFilter(data[ground_truth_key][:]))
                    psnr, ssim = calculate_metrics(y_pred, y_true)
                    results.append((full_config, psnr, ssim))

    # Handle MAT files for datasets like mice, phantom, v_phantom
    elif dataset in ["mice", "phantom", "v_phantom"]:
        path = dataset_info["path"]
        gt_file = f"{dataset}_full_recon.mat"
        config_file = os.path.join(path, full_config + ".mat")

        if not os.path.isfile(gt_file):
            print(f"[WARNING] Ground truth file not found: {gt_file}. Skipping...")
            return results

        if not os.path.isfile(config_file):
            print(f"[WARNING] Configuration file not found: {config_file}. Skipping...")
            return results

        # Load ground truth and configuration data
        gt_data = load_mat_file(gt_file, dataset_info["ground_truth"])
        config_data = load_mat_file(config_file, full_config)

        y_pred = sigMatNormalize(sigMatFilter(config_data))
        y_true = sigMatNormalize(sigMatFilter(gt_data))

        psnr, ssim = calculate_metrics(y_pred, y_true)
        results.append((full_config, psnr, ssim))

    if save_results and results:
        os.makedirs(os.path.dirname(RESULTS_DIR), exist_ok=True)
        with open(f"{RESULTS_DIR}/{dataset}_results.txt", "a") as f:
            if dataset == "MSFD":
                f.write("Configuration   Wavelength   PSNR     SSIM\n")
                f.write("---------------------------------------\n")
                for entry in results:
                    config, wavelength, psnr, ssim = entry
                    f.write(f"{config:<14} {wavelength:<11} {psnr:<7.3f} {ssim:<7.3f}\n")
            else:
                f.write("Configuration   PSNR     SSIM\n")
                f.write("-----------------------------\n")
                for entry in results:
                    config, psnr, ssim = entry
                    f.write(f"{config:<14} {psnr:<7.3f} {ssim:<7.3f}\n")
        print(f"Results saved to: {RESULTS_DIR}/{dataset}_results.txt")

    return results