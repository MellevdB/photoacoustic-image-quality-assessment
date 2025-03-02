import matplotlib.pyplot as plt
import numpy as np

# Data for MICE dataset (sparse 256 to sparse 4) with mean normalization + BandPass filter
mice_configs = ["Sparse 256", "Sparse 128", "Sparse 64", "Sparse 32", "Sparse 16", "Sparse 8", "Sparse 4"]
mice_metrics_mean_norm_filter = {
    "FSIM": ([0.877, 0.763, 0.705, 0.675, 0.658, 0.650, 0.641], [0.017, 0.018, 0.016, 0.017, 0.018, 0.021, 0.027]),
    "UQI": ([0.462, 0.183, 0.082, 0.043, 0.025, 0.015, 0.010], [0.024, 0.008, 0.005, 0.007, 0.005, 0.003, 0.003]),
    "PSNR": ([42.436, 39.271, 38.064, 37.522, 37.269, 37.147, 37.093], [2.972, 3.066, 3.103, 3.121, 3.129, 3.133, 3.135]),
    "SSIM": ([0.920, 0.823, 0.765, 0.734, 0.720, 0.713, 0.710], [0.020, 0.045, 0.060, 0.068, 0.072, 0.073, 0.074]),
    "VIF": ([0.061, 0.029, 0.016, 0.009, 0.006, 0.004, 0.004], [0.004, 0.002, 0.001, 0.001, 0.001, 0.000, 0.000]),
    "S3IM": ([0.939, 0.868, 0.821, 0.793, 0.778, 0.770, 0.767], [0.049, 0.078, 0.088, 0.091, 0.092, 0.093, 0.093])
}

# Data for PHANTOM and V_PHANTOM datasets (sparse 128 to sparse 8) with mean normalization + BandPass filter
phantom_configs = ["Sparse 128", "Sparse 64", "Sparse 32", "Sparse 16", "Sparse 8"]
phantom_metrics_mean_norm_filter = {
    "FSIM": ([0.839, 0.767, 0.706, 0.651, 0.655], [0.025, 0.030, 0.033, 0.037, 0.035]),
    "UQI": ([0.116, 0.057, 0.030, 0.016, 0.009], [0.011, 0.006, 0.004, 0.002, 0.001]),
    "PSNR": ([33.816, 29.926, 26.526, 23.024, 19.969], [2.005, 1.984, 1.973, 1.952, 1.956]),
    "SSIM": ([0.707, 0.462, 0.255, 0.123, 0.064], [0.056, 0.071, 0.071, 0.067, 0.062]),
    "VIF": ([0.017, 0.009, 0.005, 0.004, 0.003], [0.001, 0.001, 0.000, 0.000, 0.000]),
    "S3IM": ([0.698, 0.460, 0.259, 0.119, 0.056], [0.057, 0.041, 0.022, 0.010, 0.005])
}

v_phantom_metrics_mean_norm_filter = {
    "FSIM": ([0.648, 0.511, 0.432, 0.377, 0.339], [0.014, 0.023, 0.024, 0.021, 0.018]),
    "UQI": ([0.468, 0.293, 0.184, 0.118, 0.079], [0.029, 0.028, 0.022, 0.017, 0.013]),
    "PSNR": ([36.898, 30.829, 26.397, 22.628, 19.326], [2.526, 2.608, 2.640, 2.640, 2.666]),
    "SSIM": ([0.782, 0.505, 0.274, 0.136, 0.064], [0.072, 0.090, 0.072, 0.049, 0.030]),
    "VIF": ([0.104, 0.049, 0.027, 0.017, 0.012], [0.006, 0.003, 0.002, 0.001, 0.001]),
    "S3IM": ([0.792, 0.534, 0.311, 0.171, 0.094], [0.043, 0.027, 0.013, 0.008, 0.005])
}

# Zscore normalization + BandPass Filtering
mice_metrics_zscore_norm_filter = {
    "FSIM": ([0.833, 0.669, 0.576, 0.527, 0.502, 0.493, 0.492], [0.023, 0.026, 0.021, 0.017, 0.014, 0.013, 0.014]),
    "UQI": ([0.690, 0.453, 0.297, 0.212, 0.173, 0.159, 0.158], [0.045, 0.049, 0.041, 0.030, 0.023, 0.022, 0.021]),
    "PSNR": ([45.970, 41.278, 38.683, 37.126, 36.177, 35.584, 35.227], [1.582, 1.144, 0.755, 0.468, 0.287, 0.195, 0.168]),
    "SSIM": ([0.968, 0.894, 0.793, 0.699, 0.632, 0.592, 0.568], [0.010, 0.022, 0.025, 0.017, 0.013, 0.014, 0.020]),
    "VIF": ([0.849, 0.615, 0.402, 0.254, 0.167, 0.119, 0.096], [0.048, 0.083, 0.081, 0.060, 0.040, 0.029, 0.024]),
    "S3IM": ([0.812, 0.561, 0.370, 0.257, 0.198, 0.171, 0.161], [0.053, 0.079, 0.083, 0.082, 0.082, 0.081, 0.081])
}
# Zscore normalization + BandPass Filtering
phantom_metrics_zscore_norm_filter = {
    "FSIM": ([0.703, 0.589, 0.516, 0.462, 0.459], [0.026, 0.028, 0.029, 0.029, 0.026]),
    "UQI": ([0.126, 0.054, 0.032, 0.026, 0.025], [0.012, 0.006, 0.004, 0.003, 0.003]),
    "PSNR": ([36.870, 35.266, 34.313, 33.780, 33.425], [0.357, 0.218, 0.133, 0.089, 0.076]),
    "SSIM": ([0.812, 0.718, 0.651, 0.630, 0.623], [0.010, 0.008, 0.009, 0.010, 0.010]),
    "VIF": ([0.380, 0.213, 0.117, 0.074, 0.048], [0.041, 0.029, 0.018, 0.011, 0.008]),
    "S3IM": ([0.578, 0.432, 0.351, 0.329, 0.324], [0.059, 0.070, 0.073, 0.073, 0.073])
}
# Min-Max normalization (no Filtering)
v_phantom_metrics_minmax_norm_no_filter = {
    "FSIM": ([0.889, 0.826, 0.785, 0.765, 0.750], [0.021, 0.019, 0.017, 0.018, 0.020]),
    "UQI": ([0.994, 0.970, 0.922, 0.872, 0.848], [0.004, 0.024, 0.059, 0.084, 0.104]),
    "PSNR": ([31.046, 24.289, 19.782, 17.020, 16.025], [2.196, 2.180, 2.404, 2.470, 2.751]),
    "SSIM": ([0.764, 0.568, 0.441, 0.367, 0.336], [0.063, 0.074, 0.069, 0.066, 0.078]),
    "VIF": ([1.004, 0.968, 0.851, 0.704, 0.560], [0.053, 0.114, 0.190, 0.207, 0.192]),
    "S3IM": ([1.000, 0.812, 0.568, 0.369, 0.338], [0.000, 0.053, 0.074, 0.066, 0.078])
}

def plot_metrics(configs, metrics, title):
    # Reverse the configs list so that it goes from low to high
    configs_sorted = configs[::-1]
    
    # Create x positions for the sorted configurations
    x = np.arange(len(configs_sorted))
    
    # Create two subplots with equal height for both PSNR and other metrics.
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(12, 10),
                                             gridspec_kw={'height_ratios': [1, 1]})
    
    # Increase font sizes for better readability
    plt.rcParams.update({'font.size': 14})
    
    # Loop over the metrics, reversing the metric arrays so they match the sorted configs.
    for metric, (values, std) in metrics.items():
        values = np.array(values)[::-1]
        std = np.array(std)[::-1]
        
        if metric == "PSNR":
            # Plot PSNR on the top axis.
            ax_top.plot(x, values, label=metric, linestyle="--", marker="o", color="tab:blue")
            ax_top.fill_between(x, values - std, values + std, color="tab:blue", alpha=0.2)
        else:
            # Plot other metrics on the bottom axis.
            ax_bottom.plot(x, values, label=metric, linestyle="-", marker="o")
            ax_bottom.fill_between(x, values - std, values + std, alpha=0.2)
    
    # Hide the spines between axes to create a visual break.
    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)
    ax_top.xaxis.tick_top()
    ax_top.tick_params(labeltop=False)
    ax_bottom.xaxis.tick_bottom()
    
    # Add diagonal break markers.
    d = 0.015  # Size of diagonal lines.
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    
    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    
    # Set labels and ticks.
    ax_top.set_ylabel("PSNR (dB)", color="tab:blue", fontsize=18)
    ax_bottom.set_ylabel("Metric Value", fontsize=18)
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(configs_sorted, rotation=45, fontsize=16)
    
    # Add legends with larger font sizes.
    ax_top.legend(loc="upper right", fontsize=16)
    ax_bottom.legend(loc="upper left", fontsize=16)
    
    plt.suptitle(title, fontsize=20)
    plt.show()

# Plot MICE dataset
# With no title
# plot_metrics(mice_configs, mice_metrics_mean_norm_filter, "Mice Dataset")
plot_metrics(mice_configs, mice_metrics_zscore_norm_filter, "Mice Dataset")

# Plot PHANTOM dataset
# plot_metrics(phantom_configs, phantom_metrics_mean_norm_filter, "Phantom Dataset")
plot_metrics(phantom_configs, phantom_metrics_zscore_norm_filter, "Phantom Dataset")

# Plot V_PHANTOM dataset
# plot_metrics(phantom_configs, v_phantom_metrics_mean_norm_filter, "V Phantom Dataset")
plot_metrics(phantom_configs, v_phantom_metrics_minmax_norm_no_filter, "V Phantom Dataset")

