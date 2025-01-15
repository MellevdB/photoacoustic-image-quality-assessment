import numpy as np
from src.preprocessing_oadat.normalize import sigMatNormalize
from src.preprocessing_oadat.filterBandPass import sigMatFilter

def preprocess_data(sigMat, apply_filter=True, lowCutOff=0.1e6, highCutOff=6e6, fSampling=40e6, fOrder=3, conRatio=0.5):
    """
    Preprocess OADAT data by applying normalization and optional filtering.

    :param sigMat: 3D array (samples x channels x repetition) of signals
    :param apply_filter: Whether to apply bandpass filtering
    :param lowCutOff: Low cut off frequency of bandpass filter
    :param highCutOff: High cut off frequency of bandpass filter
    :param fSampling: Sampling frequency of signals
    :param fOrder: Butterworth filter order
    :param conRatio: Nyquist ratio percentage

    :return: Preprocessed 3D signal array
    """
    
    print("Shape of sigMat:", sigMat.shape)

    # Normalize the signals
    normalized_data = sigMatNormalize(sigMat)
    
    # Apply bandpass filter if needed
    if apply_filter:
        filtered_data = sigMatFilter(normalized_data, lowCutOff, highCutOff, fSampling, fOrder, conRatio)
        return filtered_data
    
    return normalized_data

if __name__ == "__main__":
    # Example usage
    dummy_data = np.random.rand(1000, 256, 10)  # Example 3D signal array
    preprocessed_data = preprocess_data(dummy_data)
    print("Preprocessed data shape:", preprocessed_data.shape)