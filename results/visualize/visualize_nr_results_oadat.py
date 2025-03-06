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

    plt.ylabel("BRISQUE Score", fontsize=18)
    plt.xticks(x, configs, rotation=45, fontsize=16)
    plt.title(title, fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.show()

# # Results with preprocessing:
# # --- Data for SCD (VC & MS) ---
# scd_configs = ["32", "64", "128"]
# scd_vc_brisque_means = [44.211, 42.149, 34.953]
# scd_vc_brisque_stds = [3.523, 4.552, 4.241]

# scd_ms_brisque_means = [61.611, 52.377, 42.047]
# scd_ms_brisque_stds = [6.394, 5.102, 4.287]

# # --- Data for SWFD (SC & MS) ---
# swfd_configs = ["32", "64", "128"]
# swfd_sc_brisque_means = [70.195, 79.288, 75.869]
# swfd_sc_brisque_stds = [6.365, 5.367, 5.704]

# swfd_ms_brisque_means = [42.484, 46.556, 44.165]
# swfd_ms_brisque_stds = [4.869, 4.406, 4.152]

# # --- Data for MSFD (w760) ---
# msfd_configs = ["32", "64", "128"]
# msfd_brisque_means = [60.229, 64.032, 62.050]
# msfd_brisque_stds = [10.130, 9.793, 8.819]

# Results without additional preprocessing:
scd_configs = ["32", "64", "128"]

scd_vc_brisque_means = [44.211, 42.149, 34.953]
scd_vc_brisque_stds = [3.523, 4.552, 4.241]

scd_ms_brisque_means = [61.611, 52.377, 42.047]
scd_ms_brisque_stds = [6.394, 5.102, 4.287]

msfd_configs = ["32", "64", "128"]

msfd_brisque_means = [60.229, 64.032, 62.050]
msfd_brisque_stds = [10.130, 9.793, 8.819]

swfd_configs = ["32", "64", "128"]

swfd_sc_brisque_means = [70.195, 79.288, 75.869]
swfd_sc_brisque_stds = [6.365, 5.367, 5.704]

swfd_ms_brisque_means = [42.484, 46.556, 44.165]
swfd_ms_brisque_stds = [4.869, 4.406, 4.152]

# --- Plot BRISQUE scores ---
plot_brisque(scd_configs, scd_vc_brisque_means, scd_vc_brisque_stds, "SCD - Virtual Circle")
plot_brisque(scd_configs, scd_ms_brisque_means, scd_ms_brisque_stds, "SCD - Multisegment")
plot_brisque(swfd_configs, swfd_sc_brisque_means, swfd_sc_brisque_stds, "SWFD - Semi Circle")
plot_brisque(swfd_configs, swfd_ms_brisque_means, swfd_ms_brisque_stds, "SWFD - Multisegment")
plot_brisque(msfd_configs, msfd_brisque_means, msfd_brisque_stds, "MSFD - Multisegment (w760)")