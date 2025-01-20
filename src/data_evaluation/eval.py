import os
import scipy.io as sio
import h5py
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from preprocessing_data.filterBandPass import sigMatFilter
from preprocessing_data.normalize import sigMatNormalize
from config.data_config import DATASETS, DATA_DIR, RESULTS_DIR

def load_mat_file(file_path, key):
    """
    Load data from a MATLAB file, supporting both v7.3 (HDF5-based) and earlier formats.

    :param file_path: Path to the `.mat` file.
    :param key: Key to extract from the file.
    :return: Data corresponding to the key.
    """
    try:
        # Try loading using scipy (non-v7.3 files)
        data = sio.loadmat(file_path)
        return data[key]
    except NotImplementedError:
        # If it's a v7.3 file, use h5py
        with h5py.File(file_path, "r") as f:
            return f[key][()]  # Extract the dataset and convert to a NumPy array


def calculate_metrics(y_pred, y_true):
    """
    Calculate PSNR and SSIM metrics using skimage's built-in functions.
    """
    data_range = y_true.max() - y_true.min()
    psnr = peak_signal_noise_ratio(y_true, y_pred, data_range=data_range)
    ssim = structural_similarity(y_true, y_pred, data_range=data_range)
    return psnr, ssim


def evaluate(dataset, config, file_key=None, save_results=True):
    """
    Evaluate a specific dataset and configuration for PSNR and SSIM.

    :param dataset: Dataset name (e.g., SCD, SWFD, MSFD, mice, phantom, v_phantom).
    :param config: Configuration name (e.g., lv128, ss32, sparse32, etc.).
    :param file_key: Optional key for datasets like SWFD (e.g., "multisegment" or "semicircle").
    :param save_results: Whether to save the results to a file.
    :return: List of tuples containing (mode/config, psnr, ssim) or (mode, wavelength, psnr, ssim) for MSFD.
    """
    results = []
    dataset_info = DATASETS[dataset]
    print(f"Processing dataset={dataset}, config={config}, file_key={file_key}")

    if dataset in ["mice", "phantom", "v_phantom"]:
        # Handle MAT files for mice, phantom, and v_phantom
        path = dataset_info["path"]
        ground_truth_key = dataset_info["ground_truth"]
        gt_file = f"{dataset}_full_recon.mat"
        gt_path = os.path.join(path, gt_file)
        gt_data = load_mat_file(gt_path, ground_truth_key)

        # Get sparse reconstructions
        file_name = f"{dataset}_{config}_recon.mat"
        file_path = os.path.join(path, file_name)
        if not os.path.isfile(file_path):
            print(f"[WARNING] File not found: {file_path}. Skipping...")
            return results

        data = load_mat_file(file_path, dataset_info["configs"][config][0])

        # Preprocess and calculate metrics
        y_pred = sigMatNormalize(sigMatFilter(data))
        y_true = sigMatNormalize(sigMatFilter(gt_data))

        psnr, ssim = calculate_metrics(y_pred, y_true)
        results.append((config, psnr, ssim))

        # Optionally save results
        if save_results:
            results_path = os.path.join(RESULTS_DIR, f"{dataset}_{config}_results.txt")
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            with open(results_path, "w") as f:
                f.write(f"Config: {config}, PSNR: {psnr:.3f}, SSIM: {ssim:.3f}\n")
            print(f"Results saved to: {results_path}")

    elif dataset in ["SCD", "SWFD", "MSFD"]:
        # Handle HDF5 files for SCD, SWFD, and MSFD
        if isinstance(dataset_info["path"], dict):
            # SWFD with multiple file keys
            data_path = os.path.join(DATA_DIR, dataset_info["path"][file_key])
            results_path = os.path.join(RESULTS_DIR, f"{dataset}_{file_key}_{config}_results.txt")
        else:
            # SCD or MSFD with a single file
            data_path = os.path.join(DATA_DIR, dataset_info["path"])
            results_path = os.path.join(RESULTS_DIR, f"{dataset}_{config}_results.txt")

        if not os.path.isfile(data_path):
            print(f"[WARNING] File not found: {data_path}. Skipping...")
            return results

        with h5py.File(data_path, "r") as data:
            modes = dataset_info["configs"][config]

            if dataset == "MSFD":
                # MSFD: Loop over wavelengths
                for mode in modes:
                    for wavelength in range(700, 851, 10):  # 700 nm to 850 nm
                        input_key = f"{mode},{config}_BP_w{wavelength}"
                        output_key = f"{mode},{config}_raw_w{wavelength}"

                        if input_key not in data or output_key not in data:
                            print(f"[SKIP] {input_key} or {output_key} not in {list(data.keys())}")
                            continue

                        y_pred = data[input_key][:]
                        y_true = data[output_key][:]
                        y_pred_f = sigMatNormalize(sigMatFilter(y_pred))
                        y_true_f = sigMatNormalize(sigMatFilter(y_true))

                        psnr, ssim = calculate_metrics(y_pred_f, y_true_f)
                        results.append((mode, wavelength, psnr, ssim))

            else:
                # SCD or SWFD: No wavelength
                for mode in modes:
                    input_key = f"{mode},{config}_BP"
                    output_key = f"{mode},{config}_raw"

                    if input_key not in data or output_key not in data:
                        print(f"[SKIP] {input_key} or {output_key} not in {list(data.keys())}")
                        continue

                    y_pred = data[input_key][:]
                    y_true = data[output_key][:]
                    y_pred_f = sigMatNormalize(sigMatFilter(y_pred))
                    y_true_f = sigMatNormalize(sigMatFilter(y_true))

                    psnr, ssim = calculate_metrics(y_pred_f, y_true_f)
                    results.append((mode, psnr, ssim))

        # Optionally save results
        if save_results and results:
            os.makedirs(os.path.dirname(results_path), exist_ok=True)
            with open(results_path, "w") as f:
                if dataset == "MSFD":
                    for mode, wavelength, psnr, ssim in results:
                        f.write(f"{mode},{config},w{wavelength},PSNR={psnr:.3f},SSIM={ssim:.3f}\n")
                else:
                    for mode, psnr, ssim in results:
                        f.write(f"{mode},{config},PSNR={psnr:.3f},SSIM={ssim:.3f}\n")
            print(f"Results saved to: {results_path}")

    return results