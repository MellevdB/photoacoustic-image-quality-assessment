import matplotlib.pyplot as plt
import numpy as np

# Configurations (Noise Levels in dB)
configs = ["10dB", "20dB", "30dB", "40dB", "50dB"]

# Denoising Data Metrics
denoising_metrics_mean_norm_filter = {
    "FSIM": ([0.304, 0.553, 0.696, 0.711, 0.713], [0.029, 0.036, 0.026, 0.027, 0.028]),
    "UQI": ([0.203, 0.313, 0.369, 0.376, 0.377], [0.130, 0.086, 0.018, 0.004, 0.004]),
    "PSNR": ([33.233, 35.693, 36.626, 36.682, 36.701], [1.773, 1.779, 1.855, 1.862, 1.864]),
    "SSIM": ([0.556, 0.738, 0.791, 0.792, 0.792], [0.178, 0.082, 0.033, 0.031, 0.031]),
    "VIF": ([0.032, 0.228, 0.370, 0.389, 0.394], [0.005, 0.026, 0.041, 0.045, 0.046]),
    "S3IM": ([0.227, 0.442, 0.529, 0.530, 0.530], [0.187, 0.161, 0.056, 0.051, 0.051])
}

denoising_metrics_zscore_norm_filter = {
    "FSIM": ([0.505, 0.733, 0.832, 0.842, 0.842], [0.028, 0.032, 0.022, 0.023, 0.023]),
    "UQI": ([0.801, 0.921, 0.933, 0.933, 0.933], [0.102, 0.056, 0.049, 0.049, 0.048]),
    "PSNR": ([19.291, 24.493, 25.261, 25.289, 25.286], [1.930, 2.176, 2.263, 2.251, 2.246]),
    "SSIM": ([0.216, 0.713, 0.829, 0.834, 0.834], [0.058, 0.064, 0.039, 0.038, 0.038]),
    "VIF": ([0.163, 0.418, 0.490, 0.498, 0.499], [0.069, 0.129, 0.147, 0.149, 0.149]),
    "S3IM": ([0.218, 0.716, 0.831, 0.836, 0.836], [0.058, 0.064, 0.038, 0.038, 0.038])
}

denoising_metrics_no_preprocessing = {
    "FSIM": ([0.316, 0.513, 0.720, 0.773, 0.776], [0.086, 0.089, 0.063, 0.042, 0.043]),
    "UQI": ([0.056, 0.120, 0.475, 0.774, 0.779], [0.095, 0.150, 0.269, 0.094, 0.092]),
    "PSNR": ([29.606, 32.099, 33.020, 33.070, 33.092], [2.717, 2.641, 2.890, 2.916, 2.923]),
    "SSIM": ([0.524, 0.807, 0.932, 0.933, 0.933], [0.246, 0.132, 0.031, 0.031, 0.031]),
    "VIF": ([0.050, 0.233, 0.382, 0.421, 0.430], [0.017, 0.061, 0.093, 0.108, 0.112]),
    "S3IM": ([0.370, 0.696, 0.927, 0.930, 0.930], [0.259, 0.196, 0.038, 0.038, 0.038])
}

def plot_metrics(title, metrics):
    x = np.arange(len(configs))
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 1]})
    plt.rcParams.update({'font.size': 14})
    
    for metric, (values, std) in metrics.items():
        values, std = np.array(values), np.array(std)
        if metric == "PSNR":
            ax_top.plot(x, values, label=metric, linestyle="--", marker="o", color="tab:blue")
            ax_top.fill_between(x, values - std, values + std, color="tab:blue", alpha=0.2)
        else:
            ax_bottom.plot(x, values, label=metric, linestyle="-", marker="o")
            ax_bottom.fill_between(x, values - std, values + std, alpha=0.2)
    
    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)
    ax_top.set_ylabel("PSNR (dB)", color="tab:blue", fontsize=18)
    ax_bottom.set_ylabel("Metric Value", fontsize=18)
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(configs, rotation=45, fontsize=16)
    ax_top.legend(loc="upper right", fontsize=16)
    ax_bottom.legend(loc="upper left", fontsize=16)
    plt.suptitle(title, fontsize=20)
    plt.show()

# Generate plot for denoising dataset
# plot_metrics("Denoising Data", denoising_metrics_mean_norm_filter)
# plot_metrics("denoising Data", denoising_metrics_zscore_norm_filter)
plot_metrics("denoising Data", denoising_metrics_no_preprocessing)