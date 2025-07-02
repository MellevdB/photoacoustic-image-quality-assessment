import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure output directory exists
output_dir = "results/zenodo"
os.makedirs(output_dir, exist_ok=True)

# X-axis labels
methods = ['method_0', 'method_1', 'method_2']
x = np.arange(len(methods))

# Metric values (mean and std)
metrics = {
    "PSNR": ([20.846, 15.8372, 12.2817], [5.0304, 5.3917, 4.6028]),
    "SSIM": ([0.7951, 0.2289, 0.4217], [0.1291, 0.1103, 0.1286]),
    "MSSSIM": ([0.8681, 0.3395, 0.2303], [0.0916, 0.1467, 0.1289]),
    "IWSSIM": ([0.7384, 0.0997, 0.0441], [0.1123, 0.0451, 0.0225]),
    # "VIF": ([0.2732, 75.6273, 1538547.75], [0.0867, 33.8496, 467393.25]),
    "FSIM": ([0.8157, 0.5856, 0.658], [0.0418, 0.0398, 0.0492]),
    "HAARPSI": ([0.3507, 0.1223, 0.0727], [0.1038, 0.0492, 0.0306]),
    "UQI": ([0.6771, 0.094, 0.1529], [0.1574, 0.0391, 0.0441]),
    "S3IM": ([0.7865, 0.1318, 0.2155], [0.1341, 0.0547, 0.0842]),
    # "TV": ([8.4382, 9.7165, 5.1973], [3.4885, 4.5466, 1.5277]),
    # "BRISQUE": ([77.6397, 63.4792, 66.2438], [14.51, 9.9383, 15.2614]),
    "CLIP-IQA": ([0.2455, 0.2422, 0.2327], [0.0765, 0.0746, 0.101])
}

# GMSD and MSGMSD separate due to different value range
separate_metrics = {
    "GMSD": ([0.1309, 0.2304, 0.2215], [0.0609, 0.0678, 0.06]),
    "MSGMSD": ([0.1369, 0.2463, 0.2377], [0.0639, 0.0655, 0.0598])
}

# Plot PSNR separately
plt.figure(figsize=(6, 4))
plt.bar(x, metrics["PSNR"][0], yerr=metrics["PSNR"][1], capsize=5, color='skyblue')
plt.xticks(x, methods)
plt.ylabel("PSNR (dB)")
plt.title("Zenodo: PSNR Comparison")
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "zenodo_psnr.png"))
plt.close()

# Plot GMSD and MSGMSD separately
for key in separate_metrics:
    plt.figure(figsize=(6, 4))
    plt.bar(x, separate_metrics[key][0], yerr=separate_metrics[key][1], capsize=5, color='lightcoral')
    plt.xticks(x, methods)
    plt.ylabel(f"{key} Score")
    plt.title(f"Zenodo: {key} Comparison")
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"zenodo_{key.lower()}.png"))
    plt.close()

# Plot all remaining metrics (skip GMSD, MSGMSD, PSNR)
plot_keys = [key for key in metrics if key not in ["PSNR", "GMSD", "MSGMSD"]]
offsets = np.linspace(-0.5, 0.5, len(plot_keys))
plt.figure(figsize=(10, 6))
bar_width = 0.08

for i, key in enumerate(plot_keys):
    means, stds = metrics[key]
    plt.bar(x + offsets[i], means, yerr=stds, width=bar_width, label=key, capsize=3)

plt.xticks(x, methods)
plt.ylabel("Metric Score")
plt.title("Zenodo: Other Metrics Comparison")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "zenodo_other_metrics.png"))
plt.close()