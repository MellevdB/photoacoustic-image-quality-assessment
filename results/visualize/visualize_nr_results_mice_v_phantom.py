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

# --- Data for MICE dataset ---
mice_configs = ["Sparse 4", "Sparse 8", "Sparse 16", "Sparse 32", "Sparse 64", "Sparse 128", "Sparse 256"]
mice_brisque_means = [48.675, 59.187, 57.535, 55.934, 56.297, 56.459, 54.093]
mice_brisque_stds = [4.666, 5.467, 4.282, 3.566, 3.336, 3.151, 3.469]

# --- Data for PHANTOM dataset ---
phantom_configs = ["Sparse 8", "Sparse 16", "Sparse 32", "Sparse 64", "Sparse 128"]
phantom_brisque_means = [76.652, 73.379, 72.226, 71.802, 73.253]
phantom_brisque_stds = [4.169, 4.081, 4.127, 3.951, 3.851]

# --- Data for V_PHANTOM dataset ---
v_phantom_brisque_means = [72.931, 75.234, 75.678, 76.071, 77.634]
v_phantom_brisque_stds = [3.050, 3.246, 3.215, 3.359, 3.498]

# --- Plot BRISQUE scores ---
plot_brisque(mice_configs, mice_brisque_means, mice_brisque_stds, "Mice Dataset")
plot_brisque(phantom_configs, phantom_brisque_means, phantom_brisque_stds, "Phantom Dataset")
plot_brisque(phantom_configs, v_phantom_brisque_means, v_phantom_brisque_stds, "V Phantom Dataset")