import matplotlib.pyplot as plt
import numpy as np

# Data for MICE dataset (sparse 256 to sparse 4)
mice_configs = ["Sparse 256", "Sparse 128", "Sparse 64", "Sparse 32", "Sparse 16", "Sparse 8", "Sparse 4"]
mice_metrics = {
    "FSIM": ([0.877, 0.763, 0.705, 0.675, 0.658, 0.650, 0.641], [0.017, 0.018, 0.016, 0.017, 0.018, 0.021, 0.027]),
    "NQM": ([0.462, 0.183, 0.082, 0.043, 0.025, 0.015, 0.010], [0.024, 0.008, 0.005, 0.007, 0.005, 0.003, 0.003]),
    "PSNR": ([42.436, 39.271, 38.064, 37.522, 37.269, 37.147, 37.093], [2.972, 3.066, 3.103, 3.121, 3.129, 3.133, 3.135]),
    "SSIM": ([0.920, 0.823, 0.765, 0.734, 0.720, 0.713, 0.710], [0.020, 0.045, 0.060, 0.068, 0.072, 0.073, 0.074]),
    "VIF": ([0.061, 0.029, 0.016, 0.009, 0.006, 0.004, 0.004], [0.004, 0.002, 0.001, 0.001, 0.001, 0.000, 0.000]),
    "S3IM": ([0.939, 0.868, 0.821, 0.793, 0.778, 0.770, 0.767], [0.049, 0.078, 0.088, 0.091, 0.092, 0.093, 0.093])
}

# Data for PHANTOM and V_PHANTOM datasets (sparse 128 to sparse 8)
phantom_configs = ["Sparse 128", "Sparse 64", "Sparse 32", "Sparse 16", "Sparse 8"]
phantom_metrics = {
    "FSIM": ([0.839, 0.767, 0.706, 0.651, 0.655], [0.025, 0.030, 0.033, 0.037, 0.035]),
    "NQM": ([0.116, 0.057, 0.030, 0.016, 0.009], [0.011, 0.006, 0.004, 0.002, 0.001]),
    "PSNR": ([33.816, 29.926, 26.526, 23.024, 19.969], [2.005, 1.984, 1.973, 1.952, 1.956]),
    "SSIM": ([0.707, 0.462, 0.255, 0.123, 0.064], [0.056, 0.071, 0.071, 0.067, 0.062]),
    "VIF": ([0.017, 0.009, 0.005, 0.004, 0.003], [0.001, 0.001, 0.000, 0.000, 0.000]),
    "S3IM": ([0.698, 0.460, 0.259, 0.119, 0.056], [0.057, 0.041, 0.022, 0.010, 0.005])
}

v_phantom_metrics = {
    "FSIM": ([0.648, 0.511, 0.432, 0.377, 0.339], [0.014, 0.023, 0.024, 0.021, 0.018]),
    "NQM": ([0.468, 0.293, 0.184, 0.118, 0.079], [0.029, 0.028, 0.022, 0.017, 0.013]),
    "PSNR": ([36.898, 30.829, 26.397, 22.628, 19.326], [2.526, 2.608, 2.640, 2.640, 2.666]),
    "SSIM": ([0.782, 0.505, 0.274, 0.136, 0.064], [0.072, 0.090, 0.072, 0.049, 0.030]),
    "VIF": ([0.104, 0.049, 0.027, 0.017, 0.012], [0.006, 0.003, 0.002, 0.001, 0.001]),
    "S3IM": ([0.792, 0.534, 0.311, 0.171, 0.094], [0.043, 0.027, 0.013, 0.008, 0.005])
}

# Function to plot metrics with y-axis rupture
def plot_metrics(configs, metrics, title):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})

    x = np.arange(len(configs))

    for metric, (values, std) in metrics.items():
        values = np.array(values)
        std = np.array(std)
        
        if metric == "PSNR":
            ax2.plot(x, values, label=metric, linestyle="--", marker="o", color="tab:blue")
            ax2.fill_between(x, values - std, values + std, color="tab:blue", alpha=0.2)
        else:
            ax1.plot(x, values, label=metric, linestyle="-", marker="o")
            ax1.fill_between(x, values - std, values + std, alpha=0.2)

    # Hide the spines between axes
    ax1.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax1.xaxis.tick_top()
    ax1.tick_params(labeltop=False)  
    ax2.xaxis.tick_bottom()

    # Add diagonal break markers
    d = .015  # Size of diagonal lines
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((-d, +d), (-d, +d), **kwargs)      
    ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  

    ax1.set_ylabel("Metric Value")
    ax2.set_ylabel("PSNR (dB)", color="tab:blue")
    ax2.set_xlabel("Configuration")
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=45)

    ax1.legend(loc="upper left", fontsize=10)
    ax2.legend(loc="upper right", fontsize=10)

    plt.suptitle(title, fontsize=14)
    plt.show()

# Plot MICE dataset
plot_metrics(mice_configs, mice_metrics, "Mice Dataset: Sparse 256 to Sparse 4")

# Plot PHANTOM dataset
plot_metrics(phantom_configs, phantom_metrics, "Phantom Dataset: Sparse 128 to Sparse 8")

# Plot V_PHANTOM dataset
plot_metrics(phantom_configs, v_phantom_metrics, "V Phantom Dataset: Sparse 128 to Sparse 8")