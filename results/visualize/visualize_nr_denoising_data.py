import numpy as np
import matplotlib.pyplot as plt

# Configurations (Noise Levels in dB)
configs = ["10dB", "20dB", "30dB", "40dB", "50dB"]

# BRISQUE Scores for Denoising Data
denoising_brisque_mean_norm_filter = {
    "BRISQUE": ([108.570, 93.643, 68.554, 34.757, 33.191], [2.640, 3.888, 8.217, 4.966, 4.573])
}

denoising_brisque_zscore_norm_filter = {
    "BRISQUE": ([89.886, 61.056, 37.271, 58.534, 59.356], [3.528, 6.399, 9.369, 6.698, 6.545])
}

def plot_brisque(configs, brisque_means, brisque_stds, title):
    """
    Creates a line plot for BRISQUE scores with error bands (std deviation).
    
    :param configs: X-axis labels (configurations)
    :param brisque_means: Mean BRISQUE values
    :param brisque_stds: Standard deviation of BRISQUE scores
    :param title: Title of the graph
    """
    x = np.arange(len(configs))
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, brisque_means, label="BRISQUE", color="r", linestyle="-", marker="o")
    plt.fill_between(x, np.array(brisque_means) - np.array(brisque_stds), 
                     np.array(brisque_means) + np.array(brisque_stds), color="r", alpha=0.2)

    plt.ylabel("BRISQUE Score", fontsize=18)
    plt.xticks(x, configs, rotation=45, fontsize=14)
    plt.title(title, fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.show()

# Generate BRISQUE plot for denoising dataset
# plot_brisque(configs, denoising_brisque_mean_norm_filter["BRISQUE"][0], denoising_brisque_mean_norm_filter["BRISQUE"][1], "Denoising Data")
plot_brisque(configs, denoising_brisque_zscore_norm_filter["BRISQUE"][0], denoising_brisque_zscore_norm_filter["BRISQUE"][1], "Denoising Data")