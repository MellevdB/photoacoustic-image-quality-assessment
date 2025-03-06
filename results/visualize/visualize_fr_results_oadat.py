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
    # Reverse the configs list to order from low to high
    configs_sorted = configs[::-1]
    x = np.arange(len(configs_sorted))
    
    # Create two subplots with equal heights (PSNR on top, others on bottom)
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(10, 8), 
                                              gridspec_kw={'height_ratios': [1, 1]})
    
    # Define colors for each metric
    colors = {
        "FSIM": "b",
        "UQI": "g",
        "PSNR": "r",
        "SSIM": "c",
        "VIF": "m",
        "S3IM": "y"
    }
    
    # Loop over each metric, reverse the order of values to match configs_sorted
    for metric, mean_values in metric_means.items():
        std_values = metric_stds[metric]
        mean_values = np.array(mean_values)[::-1]
        std_values = np.array(std_values)[::-1]
        color = colors.get(metric, "k")
        
        if metric == "PSNR":
            # Plot PSNR on the top axis
            ax_top.plot(x, mean_values, label=metric, color=color, linestyle="dashed", marker="o")
            ax_top.fill_between(x, mean_values - std_values, mean_values + std_values, color=color, alpha=0.2)
        else:
            # Plot other metrics on the bottom axis
            ax_bottom.plot(x, mean_values, label=metric, color=color, marker="o")
            ax_bottom.fill_between(x, mean_values - std_values, mean_values + std_values, color=color, alpha=0.2)
    
    # Hide the spines between axes for a visual break
    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)
    
    # Set ticks: top axis gets the top ticks, bottom axis gets the bottom ticks.
    ax_top.xaxis.tick_top()
    ax_top.tick_params(labeltop=False)
    ax_bottom.xaxis.tick_bottom()
    
    # Add diagonal break markers
    d = .015  # Size of diagonal lines
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1-d, 1+d), (-d, +d), **kwargs)
    
    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1-d, 1+d), **kwargs)
    ax_bottom.plot((1-d, 1+d), (1-d, 1+d), **kwargs)
    
    # Set labels and ticks
    ax_top.set_ylabel("PSNR (dB)", color="r", fontsize=18)
    ax_bottom.set_ylabel("Metric Value", fontsize=18)
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(configs_sorted, rotation=45, fontsize=16)
    
    # Add legends
    ax_top.legend(loc="upper right", fontsize=14)
    ax_bottom.legend(loc="upper left", fontsize=14)
    
    plt.suptitle(title, fontsize=20)
    plt.show()


# # Results with preprocessing (was not necessary)
# # --- Data for SCD (VC & MS) ---
# scd_configs = ["128", "64", "32"]
# scd_vc_means = {
#     "FSIM": [0.806, 0.701, 0.640],
#     "UQI": [0.063, 0.033, 0.024],
#     "PSNR": [27.918, 27.326, 27.050],
#     "SSIM": [0.526, 0.478, 0.458],
#     "VIF": [0.019, 0.009, 0.005],
#     "S3IM": [0.652, 0.596, 0.568]
# }
# scd_vc_stds = {
#     "FSIM": [0.021, 0.024, 0.027],
#     "UQI": [0.003, 0.005, 0.004],
#     "PSNR": [1.779, 1.783, 1.790],
#     "SSIM": [0.059, 0.065, 0.067],
#     "VIF": [0.002, 0.001, 0.000],
#     "S3IM": [0.050, 0.055, 0.057]
# }
# scd_ms_means = {
#     "FSIM": [0.973, 0.852, 0.742],
#     "UQI": [0.483, 0.158, 0.053],
#     "PSNR": [34.393, 30.880, 29.574],
#     "SSIM": [0.861, 0.687, 0.597],
#     "VIF": [0.047, 0.024, 0.011],
#     "S3IM": [0.901, 0.766, 0.684]
# }
# scd_ms_stds = {
#     "FSIM": [0.005, 0.013, 0.021],
#     "UQI": [0.015, 0.007, 0.002],
#     "PSNR": [1.714, 1.713, 1.719],
#     "SSIM": [0.022, 0.048, 0.060],
#     "VIF": [0.003, 0.002, 0.001],
#     "S3IM": [0.020, 0.043, 0.052]
# }

# # --- Data for SWFD (SC) ---
# swfd_configs = ["128", "64", "32"]
# swfd_sc_means = {
#     "FSIM": [0.951, 0.845, 0.766],
#     "UQI": [0.365, 0.137, 0.055],
#     "PSNR": [43.402, 40.582, 39.475],
#     "SSIM": [0.956, 0.916, 0.893],
#     "VIF": [0.035, 0.017, 0.009],
#     "S3IM": [0.968, 0.940, 0.922]
# }
# swfd_sc_stds = {
#     "FSIM": [0.006, 0.013, 0.019],
#     "UQI": [0.013, 0.005, 0.002],
#     "PSNR": [2.090, 2.101, 2.115],
#     "SSIM": [0.012, 0.022, 0.027],
#     "VIF": [0.002, 0.001, 0.001],
#     "S3IM": [0.029, 0.048, 0.058]
# }

# swfd_ms_means = {
#     "FSIM": [0.957, 0.863, 0.771],
#     "UQI": [0.420, 0.156, 0.059],
#     "PSNR": [38.220, 35.195, 34.008],
#     "SSIM": [0.887, 0.775, 0.712],
#     "VIF": [0.040, 0.020, 0.010],
#     "S3IM": [0.916, 0.830, 0.776]
# }
# swfd_ms_stds = {
#     "FSIM": [0.005, 0.012, 0.019],
#     "UQI": [0.013, 0.006, 0.003],
#     "PSNR": [2.075, 2.128, 2.145],
#     "SSIM": [0.024, 0.047, 0.060],
#     "VIF": [0.002, 0.001, 0.001],
#     "S3IM": [0.034, 0.057, 0.068]
# }

# # --- Data for MSFD (w760) ---
# msfd_configs = ["128", "64", "32"]
# msfd_means = {
#     "FSIM": [0.957, 0.869, 0.793],
#     "UQI": [0.384, 0.139, 0.053],
#     "PSNR": [38.053, 35.040, 33.905],
#     "SSIM": [0.898, 0.801, 0.752],
#     "VIF": [0.040, 0.019, 0.009],
#     "S3IM": [0.916, 0.838, 0.795]
# }
# msfd_stds = {
#     "FSIM": [0.005, 0.015, 0.023],
#     "UQI": [0.011, 0.006, 0.003],
#     "PSNR": [2.472, 2.486, 2.489],
#     "SSIM": [0.031, 0.056, 0.066],
#     "VIF": [0.002, 0.001, 0.001],
#     "S3IM": [0.047, 0.076, 0.086]
# }

# Results without additional preprocessing
scd_configs = ["128", "64", "32"]

scd_vc_means = {
    "FSIM": [0.910, 0.833, 0.758],
    "UQI": [0.195, 0.127, 0.095],
    "PSNR": [28.899, 24.508, 22.162],
    "SSIM": [0.506, 0.279, 0.189],
    "VIF": [0.939, 0.797, 0.601],
    "S3IM": [0.502, 0.278, 0.189]
}
scd_vc_stds = {
    "FSIM": [0.024, 0.025, 0.026],
    "UQI": [0.040, 0.030, 0.022],
    "PSNR": [1.591, 1.310, 1.110],
    "SSIM": [0.091, 0.074, 0.059],
    "VIF": [0.074, 0.146, 0.188],
    "S3IM": [0.090, 0.074, 0.059]
}

scd_ms_means = {
    "FSIM": [0.931, 0.810, 0.752],
    "UQI": [0.481, 0.242, 0.136],
    "PSNR": [38.661, 29.554, 24.161],
    "SSIM": [0.895, 0.623, 0.398],
    "VIF": [0.986, 0.939, 0.824],
    "S3IM": [0.893, 0.620, 0.398]
}
scd_ms_stds = {
    "FSIM": [0.010, 0.024, 0.035],
    "UQI": [0.051, 0.040, 0.028],
    "PSNR": [1.697, 1.423, 0.924],
    "SSIM": [0.032, 0.076, 0.085],
    "VIF": [0.026, 0.060, 0.084],
    "S3IM": [0.032, 0.075, 0.084]
}

msfd_configs = ["128", "64", "32"]

msfd_means = {
    "FSIM": [0.968, 0.906, 0.849],
    "UQI": [0.443, 0.215, 0.100],
    "PSNR": [33.929, 28.086, 24.321],
    "SSIM": [0.788, 0.508, 0.287],
    "VIF": [0.991, 0.981, 0.917],
    "S3IM": [0.786, 0.506, 0.285]
}
msfd_stds = {
    "FSIM": [0.005, 0.016, 0.029],
    "UQI": [0.027, 0.021, 0.014],
    "PSNR": [2.250, 2.162, 1.703],
    "SSIM": [0.047, 0.068, 0.054],
    "VIF": [0.054, 0.090, 0.193],
    "S3IM": [0.048, 0.068, 0.054]
}

swfd_configs = ["128", "64", "32"]

swfd_sc_means = {
    "FSIM": [0.965, 0.892, 0.833],
    "UQI": [0.447, 0.211, 0.092],
    "PSNR": [37.781, 32.137, 27.965],
    "SSIM": [0.866, 0.661, 0.410],
    "VIF": [1.005, 1.037, 1.069],
    "S3IM": [0.864, 0.660, 0.409]
}
swfd_sc_stds = {
    "FSIM": [0.007, 0.016, 0.026],
    "UQI": [0.025, 0.018, 0.012],
    "PSNR": [2.602, 2.379, 2.086],
    "SSIM": [0.056, 0.090, 0.081],
    "VIF": [0.025, 0.056, 0.109],
    "S3IM": [0.056, 0.090, 0.081]
}

swfd_ms_means = {
    "FSIM": [0.960, 0.888, 0.828],
    "UQI": [0.503, 0.253, 0.111],
    "PSNR": [35.165, 29.589, 25.254],
    "SSIM": [0.804, 0.523, 0.256],
    "VIF": [0.997, 0.998, 1.010],
    "S3IM": [0.802, 0.520, 0.254]
}
swfd_ms_stds = {
    "FSIM": [0.006, 0.014, 0.020],
    "UQI": [0.037, 0.031, 0.016],
    "PSNR": [2.068, 1.889, 1.667],
    "SSIM": [0.048, 0.071, 0.050],
    "VIF": [0.042, 0.064, 0.112],
    "S3IM": [0.049, 0.071, 0.050]
}


# --- Plotting ---
plot_metrics(scd_configs, scd_vc_means, scd_vc_stds, "SCD - Virtual Circle")
plot_metrics(scd_configs, scd_ms_means, scd_ms_stds, "SCD - Multisegment")
plot_metrics(swfd_configs, swfd_sc_means, swfd_sc_stds, "SWFD - Semi Circle")
plot_metrics(swfd_configs, swfd_ms_means, swfd_ms_stds, "SWFD - Multisegment")
plot_metrics(msfd_configs, msfd_means, msfd_stds, "MSFD - Multisegment (w760)")