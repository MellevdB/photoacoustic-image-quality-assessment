import os
import numpy as np
import h5py
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# If your actual preprocessing differs, adjust these imports:
from preprocessing_oadat.filterBandPass import sigMatFilter
from preprocessing_oadat.normalize import sigMatNormalize

from config.data_config import OADAT_DATA_DIR, RESULTS_DIR, DATASETS


def calculate_metrics(y_pred, y_true):
    """
    Calculate PSNR and SSIM metrics using skimage's built-in functions.
    """
    # Compute data_range from the ground truth
    data_range = y_true.max() - y_true.min()
    print(f"data shape y true = {y_true.shape}, data shape y pred = {y_pred.shape}")
    psnr_value = peak_signal_noise_ratio(y_true, y_pred, data_range=data_range)
    ssim_value = structural_similarity(y_true, y_pred, data_range=data_range)
    return psnr_value, ssim_value


def evaluate(dataset, geometry, file_key=None, save_results=True):
    """
    Evaluate a dataset+geometry (e.g., "SCD" + "lv128") by iterating 
    over the 'modes' (e.g., ["li", "vc", "ms"]) found in DATASETS[dataset]["configs"][geometry].

    For MSFD, we loop over wavelengths 700..850 and look for
    'mode,geometry_BP_wXXX' and 'mode,geometry_raw_wXXX'.
    For SCD/SWFD, just use 'mode,geometry_BP' / 'mode,geometry_raw'.

    :param dataset: One of ["SCD", "SWFD", "MSFD"]
    :param geometry: e.g., "lv128", "ss128", "ss64" ...
    :param file_key: for SWFD, can be "multisegment" or "semicircle"; for SCD/MSFD usually None
    :param save_results: whether to save the results to a text file
    :return: list of tuples (mode, psnr, ssim) or (mode, wavelength, psnr, ssim) if MSFD
    """
    results = []
    print(f"Processing dataset={dataset}, geometry={geometry}, file_key={file_key}")

    # Figure out correct HDF5 path:
    if isinstance(DATASETS[dataset]["path"], dict):
        # e.g., SWFD with multiple file keys
        data_path = os.path.join(OADAT_DATA_DIR, DATASETS[dataset]["path"][file_key])
        out_filename = f"{dataset}_{file_key}_{geometry}_results.txt"
    else:
        # e.g., SCD or MSFD with single file
        data_path = os.path.join(OADAT_DATA_DIR, DATASETS[dataset]["path"])
        out_filename = f"{dataset}_{geometry}_results.txt"

    results_path = os.path.join(RESULTS_DIR, out_filename)

    if not os.path.isfile(data_path):
        print(f"[WARNING] File not found: {data_path}. Skipping...")
        return results

    # Open the HDF5 file
    with h5py.File(data_path, "r") as data:
        # Grab all 'modes' for this geometry
        modes = DATASETS[dataset]["configs"][geometry]

        # If it's MSFD, do a wavelength loop. Otherwise, no loop.
        if dataset == "MSFD":
            # Summarize results for each wavelength from 700 to 850 in steps of 10
            for mode in modes:
                for wavelength in range(700, 851, 10):
                    input_key = f"{mode},{geometry}_BP_w{wavelength}"
                    output_key = f"{mode},{geometry}_raw_w{wavelength}"

                    if input_key not in data or output_key not in data:
                        print(f"[SKIP] {input_key} or {output_key} not in {list(data.keys())}")
                        continue
                    else:
                        print(f"[INFO] Found keys: {input_key} and {output_key}, processing...")

                    # Load
                    y_pred = data[input_key][:]
                    y_true = data[output_key][:]

                    # Preprocess: filter + normalize
                    y_pred_f = sigMatNormalize(sigMatFilter(y_pred))
                    y_true_f = sigMatNormalize(sigMatFilter(y_true))

                    # Compute metrics
                    psnr_val, ssim_val = calculate_metrics(y_pred_f, y_true_f)

                    results.append((mode, wavelength, psnr_val, ssim_val))

        else:
            # SCD or SWFD: no wavelength dimension
            for mode in modes:
                input_key = f"{mode},{geometry}_BP"
                output_key = f"{mode},{geometry}_raw"

                if input_key not in data or output_key not in data:
                    print(f"[SKIP] {input_key} or {output_key} not in {list(data.keys())}")
                    continue
                else:
                    print(f"[INFO] Found keys: {input_key} and {output_key}, processing...")

                y_pred = data[input_key][:]
                y_true = data[output_key][:]

                y_pred_f = sigMatNormalize(sigMatFilter(y_pred))
                y_true_f = sigMatNormalize(sigMatFilter(y_true))

                psnr_val, ssim_val = calculate_metrics(y_pred_f, y_true_f)

                # We'll store (mode, psnr, ssim) here
                results.append((mode, psnr_val, ssim_val))

    # Optionally save to file
    if save_results and results:
        with open(results_path, "w") as f:
            if dataset == "MSFD":
                # Write lines including wavelength
                for (mode, wavelength, psnr_val, ssim_val) in results:
                    f.write(f"{mode},{geometry},w{wavelength},PSNR={psnr_val:.3f},SSIM={ssim_val:.3f}\n")
            else:
                # No wavelength
                for (mode, psnr_val, ssim_val) in results:
                    f.write(f"{mode},{geometry},PSNR={psnr_val:.3f},SSIM={ssim_val:.3f}\n")

        print(f"Results saved to: {results_path}")

    # Print to console for clarity
    if dataset == "MSFD":
        for (mode, wavelength, psnr_val, ssim_val) in results:
            print(f"[{dataset}] geometry={geometry}, mode={mode}, w={wavelength}: "
                  f"PSNR={psnr_val:.3f}, SSIM={ssim_val:.3f}")
    else:
        for (mode, psnr_val, ssim_val) in results:
            print(f"[{dataset}] geometry={geometry}, mode={mode}: "
                  f"PSNR={psnr_val:.3f}, SSIM={ssim_val:.3f}")

    return results