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

    plt.xlabel("Configurations")
    plt.ylabel("BRISQUE Score")
    plt.xticks(x, configs, rotation=45)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.show()

# --- Data for SCD (VC & MS) ---
scd_configs = ["128", "64", "32"]
scd_vc_brisque_means = [34.953, 42.149, 44.211]
scd_vc_brisque_stds = [4.241, 4.552, 3.523]

scd_ms_brisque_means = [42.047, 52.377, 61.611]
scd_ms_brisque_stds = [4.287, 5.102, 6.394]

# --- Data for SWFD (SC & MS) ---
swfd_configs = ["128", "64", "32"]
swfd_sc_brisque_means = [75.869, 79.288, 70.195]
swfd_sc_brisque_stds = [5.704, 5.367, 6.365]

swfd_ms_brisque_means = [44.165, 46.556, 42.484]
swfd_ms_brisque_stds = [4.152, 4.406, 4.869]

# --- Data for MSFD (w760) ---
msfd_configs = ["128", "64", "32"]
msfd_brisque_means = [62.050, 64.032, 60.229]
msfd_brisque_stds = [8.819, 9.793, 10.130]

# --- Plot BRISQUE scores ---
plot_brisque(scd_configs, scd_vc_brisque_means, scd_vc_brisque_stds, "SCD - Virtual Circle")
plot_brisque(scd_configs, scd_ms_brisque_means, scd_ms_brisque_stds, "SCD - Multisegment")
plot_brisque(swfd_configs, swfd_sc_brisque_means, swfd_sc_brisque_stds, "SWFD - Semi Circle")
plot_brisque(swfd_configs, swfd_ms_brisque_means, swfd_ms_brisque_stds, "SWFD - Multisegment")
plot_brisque(msfd_configs, msfd_brisque_means, msfd_brisque_stds, "MSFD - Multisegment (w760)")