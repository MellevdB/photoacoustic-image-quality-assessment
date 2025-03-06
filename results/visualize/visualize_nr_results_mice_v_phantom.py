import numpy as np
import matplotlib.pyplot as plt

# --- Function to plot BRISQUE scores ---
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

    plt.xlabel("Configurations", fontsize=18)
    plt.ylabel("BRISQUE Score", fontsize=18)
    plt.xticks(x, configs, rotation=45, fontsize=16)
    plt.title(title, fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.show()

# # --- Data for MICE dataset ---
mice_configs = ["Sparse 4", "Sparse 8", "Sparse 16", "Sparse 32", "Sparse 64", "Sparse 128", "Sparse 256"]
mice_brisque_means_mean_norm_filter = [48.675, 59.187, 57.535, 55.934, 56.297, 56.459, 54.093]
mice_brisque_stds_mean_norm_filter = [4.666, 5.467, 4.282, 3.566, 3.336, 3.151, 3.469]

# # --- Data for PHANTOM dataset ---
phantom_configs = ["Sparse 8", "Sparse 16", "Sparse 32", "Sparse 64", "Sparse 128"]
phantom_brisque_means_mean_norm_filter = [76.652, 73.379, 72.226, 71.802, 73.253]
phantom_brisque_stds_mean_norm_filter = [4.169, 4.081, 4.127, 3.951, 3.851]

# --- Data for V_PHANTOM dataset ---
v_phantom_brisque_means_mean_norm_filter = [72.931, 75.234, 75.678, 76.071, 77.634]
v_phantom_brisque_stds_mean_norm_filter = [3.050, 3.246, 3.215, 3.359, 3.498]



mice_brisque_means_zscore_norm_filter = [47.058, 57.970, 56.466, 54.960, 55.245, 55.164, 52.317]
mice_brisque_stds_zscore_norm_filter = [4.344, 5.235, 4.091, 3.455, 3.226, 3.088, 3.381]

phantom_brisque_means_zscore_norm_filter = [75.784, 72.464, 71.246, 70.753, 72.139]
phantom_brisque_stds_zscore_norm_filter = [4.331, 4.269, 4.342, 4.157, 4.078]

v_phantom_brisque_means_minmax_norm_no_filter = [76.358, 77.992, 77.524, 77.686, 78.500]
v_phantom_brisque_stds_minmax_norm_no_filter = [3.424, 4.451, 4.532, 4.190, 4.220]

# Results no additional preprocessing
mice_brisque_no_preprocessing = [60.247, 60.991, 60.625, 60.558, 62.477, 63.937, 54.372], [4.125, 3.837, 3.860, 3.813, 4.315, 5.421, 6.144]
phantom_brisque_no_preprocessing = [74.080, 72.377, 73.247, 75.263, 79.256], [5.895, 6.118, 6.243, 5.769, 5.586]
v_phantom_brisque_no_preprocessing = [82.698, 79.448, 78.079, 77.584, 74.419], [6.363, 5.957, 6.097, 5.652, 4.512]

# --- Plot BRISQUE scores ---
# Mean normalization + BandPass Filter
# plot_brisque(mice_configs, mice_brisque_means_mean_norm_filter, mice_brisque_stds_mean_norm_filter, "Mice Dataset")
# plot_brisque(phantom_configs, phantom_brisque_means_mean_norm_filter, phantom_brisque_stds_mean_norm_filter, "Phantom Dataset")
# plot_brisque(phantom_configs, v_phantom_brisque_means_mean_norm_filter, v_phantom_brisque_stds_mean_norm_filter, "V Phantom Dataset")

# Zscore normalization + BandPass filter for mice and phantom
# MinMax normalization (no filter) for v_phantom
# plot_brisque(mice_configs, mice_brisque_means_zscore_norm_filter, mice_brisque_stds_zscore_norm_filter, "Mice Dataset")
# plot_brisque(phantom_configs, phantom_brisque_means_zscore_norm_filter, phantom_brisque_stds_zscore_norm_filter, "Phantom Dataset")
# plot_brisque(phantom_configs, v_phantom_brisque_means_minmax_norm_no_filter, v_phantom_brisque_stds_minmax_norm_no_filter, "V Phantom Dataset")

# Plot results no additional preprocessing
plot_brisque(mice_configs, mice_brisque_no_preprocessing[0], mice_brisque_no_preprocessing[1], "Mice Dataset")
plot_brisque(phantom_configs, phantom_brisque_no_preprocessing[0], phantom_brisque_no_preprocessing[1], "Phantom Dataset")
plot_brisque(phantom_configs, v_phantom_brisque_no_preprocessing[0], v_phantom_brisque_no_preprocessing[1], "V Phantom Dataset")