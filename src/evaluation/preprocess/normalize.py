#-----
# Description   : Normalization around mean value of each channel
# Date          : February 2021
# Author        : Berkan Lafci
# E-mail        : lafciberkan@gmail.com
#-----

# Added own code for different normalization methods

import logging
import time
import numpy as np

def sigMatNormalize(sigMatIn, method="mean"):
    """
    Normalizes the input signal matrix using different methods.
    
    :param sigMatIn: 3D array (slices x height x width).
    :param method: "mean" (OADAT-style), "zscore" (for high variance), "minmax" (for low variance), "log" (for contrast).
    :return: Normalized 3D array.
    """
    logging.info(f'Function "sigMatNormalize" using {method} normalization')

    print(f'***** Applying {method} normalization *****')
    startTime = time.time()

    sigMatOut = np.zeros_like(sigMatIn)

    if method == "mean":
        # ✅ OADAT-style mean normalization
        for i in range(sigMatIn.shape[2]):  # Loop over slices
            singleF = sigMatIn[:, :, i]
            meanF = np.mean(singleF, axis=0)  # Mean per channel
            sigMatOut[:, :, i] = singleF - np.tile(meanF, (singleF.shape[0], 1))  # Normalize
        
    elif method == "zscore":
        # ✅ Standard Z-score normalization
        mean = np.mean(sigMatIn, axis=(1, 2), keepdims=True)
        std = np.std(sigMatIn, axis=(1, 2), keepdims=True) + 1e-8  # Avoid division by zero
        sigMatOut = (sigMatIn - mean) / std
    
    elif method == "minmax":
        # ✅ Min-Max normalization
        min_val = np.min(sigMatIn, axis=(1, 2), keepdims=True)
        max_val = np.max(sigMatIn, axis=(1, 2), keepdims=True)
        sigMatOut = (sigMatIn - min_val) / (max_val - min_val + 1e-8)

    elif method == "log":
        # ✅ Log transformation (for extreme range compression)
        sigMatIn_shifted = sigMatIn - np.min(sigMatIn) + 1e-8  # Shift to positive range
        sigMatOut = np.log1p(sigMatIn_shifted)  # Apply log transformation

    else:
        raise ValueError("Unknown normalization method. Choose from 'mean', 'zscore', 'minmax', 'log'.")

    endTime = time.time()
    print(f'Time elapsed: {endTime - startTime:.2f} sec')

    return sigMatOut