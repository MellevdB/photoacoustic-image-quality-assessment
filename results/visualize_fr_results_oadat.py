import numpy as np
import matplotlib.pyplot as plt

# --- Function to plot metrics with y-axis rupture ---
def plot_metrics(configs, metric_means, metric_stds, title):
    """
    Creates a line plot for multiple metrics with a ruptured y-axis.
    
    :param configs: X-axis labels (configurations)
    :param metric_means: Dictionary of metric means
    :param metric_stds: Dictionary of metric standard deviations
    :param title: Title of the graph
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8), gridspec_kw={'height_ratios': [1, 2]})

    x = np.arange(len(configs))
    colors = {
        "FSIM": "b",
        "NQM": "g",
        "PSNR": "r",
        "SSIM": "c",
        "VIF": "m",
        "S3IM": "y"
    }

    for metric, mean_values in metric_means.items():
        std_values = metric_stds[metric]
        color = colors[metric]

        if metric == "PSNR":
            ax1.plot(x, mean_values, label=metric, color=color, linestyle="dashed", marker="o")
            ax1.fill_between(x, np.array(mean_values) - np.array(std_values), 
                             np.array(mean_values) + np.array(std_values), color=color, alpha=0.2)
        else:
            ax2.plot(x, mean_values, label=metric, color=color, marker="o")
            ax2.fill_between(x, np.array(mean_values) - np.array(std_values), 
                             np.array(mean_values) + np.array(std_values), color=color, alpha=0.2)

    # Hide spines between axes
    ax2.spines['top'].set_visible(False)
    ax1.spines['bottom'].set_visible(False)
    ax2.xaxis.tick_top()
    ax2.tick_params(labeltop=False)  
    ax1.xaxis.tick_bottom()

    # Add diagonal break markers
    d = .015  # Size of diagonal lines
    kwargs = dict(transform=ax2.transAxes, color='k', clip_on=False)
    ax2.plot((-d, +d), (-d, +d), **kwargs)      
    ax2.plot((1 - d, 1 + d), (-d, +d), **kwargs)

    kwargs.update(transform=ax1.transAxes)
    ax1.plot((-d, +d), (1 - d, 1 + d), **kwargs)  
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  

    ax2.set_ylabel("Metric Values (FSIM, NQM, SSIM, VIF, S3IM)")
    ax1.set_ylabel("PSNR (dB)", color="tab:red")
    ax1.set_xlabel("Configurations")
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=45)

    ax2.legend(loc="upper left", fontsize=10)
    ax1.legend(loc="upper right", fontsize=10)

    plt.suptitle(title, fontsize=14)
    plt.show()


# --- Data for SCD (VC & MS) ---
scd_configs = ["128", "64", "32"]
scd_vc_means = {
    "FSIM": [0.806, 0.701, 0.640],
    "NQM": [0.063, 0.033, 0.024],
    "PSNR": [27.918, 27.326, 27.050],
    "SSIM": [0.526, 0.478, 0.458],
    "VIF": [0.019, 0.009, 0.005],
    "S3IM": [0.652, 0.596, 0.568]
}
scd_vc_stds = {
    "FSIM": [0.021, 0.024, 0.027],
    "NQM": [0.003, 0.005, 0.004],
    "PSNR": [1.779, 1.783, 1.790],
    "SSIM": [0.059, 0.065, 0.067],
    "VIF": [0.002, 0.001, 0.000],
    "S3IM": [0.050, 0.055, 0.057]
}
scd_ms_means = {
    "FSIM": [0.973, 0.852, 0.742],
    "NQM": [0.483, 0.158, 0.053],
    "PSNR": [34.393, 30.880, 29.574],
    "SSIM": [0.861, 0.687, 0.597],
    "VIF": [0.047, 0.024, 0.011],
    "S3IM": [0.901, 0.766, 0.684]
}
scd_ms_stds = {
    "FSIM": [0.005, 0.013, 0.021],
    "NQM": [0.015, 0.007, 0.002],
    "PSNR": [1.714, 1.713, 1.719],
    "SSIM": [0.022, 0.048, 0.060],
    "VIF": [0.003, 0.002, 0.001],
    "S3IM": [0.020, 0.043, 0.052]
}

plot_metrics(scd_configs, scd_vc_means, scd_vc_stds, "SCD - Virtual Circle")
plot_metrics(scd_configs, scd_ms_means, scd_ms_stds, "SCD - Multisegment")

# --- Data for SWFD (SC) ---
swfd_configs = ["128", "64", "32"]
swfd_sc_means = {
    "FSIM": [0.951, 0.845, 0.766],
    "NQM": [0.365, 0.137, 0.055],
    "PSNR": [43.402, 40.582, 39.475],
    "SSIM": [0.956, 0.916, 0.893],
    "VIF": [0.035, 0.017, 0.009],
    "S3IM": [0.968, 0.940, 0.922]
}
swfd_sc_stds = {
    "FSIM": [0.006, 0.013, 0.019],
    "NQM": [0.013, 0.005, 0.002],
    "PSNR": [2.090, 2.101, 2.115],
    "SSIM": [0.012, 0.022, 0.027],
    "VIF": [0.002, 0.001, 0.001],
    "S3IM": [0.029, 0.048, 0.058]
}

plot_metrics(swfd_configs, swfd_sc_means, swfd_sc_stds, "SWFD - Semi Circle")

# --- Data for MSFD (w760) ---
msfd_configs = ["128", "64", "32"]
msfd_means = {
    "FSIM": [0.957, 0.869, 0.793],
    "NQM": [0.384, 0.139, 0.053],
    "PSNR": [38.053, 35.040, 33.905],
    "SSIM": [0.898, 0.801, 0.752],
    "VIF": [0.040, 0.019, 0.009],
    "S3IM": [0.916, 0.838, 0.795]
}
msfd_stds = {
    "FSIM": [0.005, 0.015, 0.023],
    "NQM": [0.011, 0.006, 0.003],
    "PSNR": [2.472, 2.486, 2.489],
    "SSIM": [0.031, 0.056, 0.066],
    "VIF": [0.002, 0.001, 0.001],
    "S3IM": [0.047, 0.076, 0.086]
}

plot_metrics(msfd_configs, msfd_means, msfd_stds, "MSFD - Multisegment (w760)")