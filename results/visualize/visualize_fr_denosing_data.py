import matplotlib.pyplot as plt
import numpy as np

# Configurations (Noise Levels in dB)
configs = ["10dB", "20dB", "30dB", "40dB", "50dB"]

# Denoising Data Metrics
denoising_metrics = {
    "FSIM": ([0.304, 0.553, 0.696, 0.711, 0.713], [0.029, 0.036, 0.026, 0.027, 0.028]),
    "UQI": ([0.203, 0.313, 0.369, 0.376, 0.377], [0.130, 0.086, 0.018, 0.004, 0.004]),
    "PSNR": ([33.233, 35.693, 36.626, 36.682, 36.701], [1.773, 1.779, 1.855, 1.862, 1.864]),
    "SSIM": ([0.556, 0.738, 0.791, 0.792, 0.792], [0.178, 0.082, 0.033, 0.031, 0.031]),
    "VIF": ([0.032, 0.228, 0.370, 0.389, 0.394], [0.005, 0.026, 0.041, 0.045, 0.046]),
    "S3IM": ([0.227, 0.442, 0.529, 0.530, 0.530], [0.187, 0.161, 0.056, 0.051, 0.051])
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
plot_metrics("Denoising Data", denoising_metrics)