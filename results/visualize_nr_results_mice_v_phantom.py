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

# --- Data for MICE dataset ---
mice_configs = ["Sparse 256", "Sparse 128", "Sparse 64", "Sparse 32", "Sparse 16", "Sparse 8", "Sparse 4"]
mice_brisque_means = [54.093, 56.459, 56.297, 55.934, 57.535, 59.187, 48.675]
mice_brisque_stds = [3.469, 3.151, 3.336, 3.566, 4.282, 5.467, 4.666]

# --- Data for PHANTOM dataset ---
phantom_configs = ["Sparse 128", "Sparse 64", "Sparse 32", "Sparse 16", "Sparse 8"]
phantom_brisque_means = [73.253, 71.802, 72.226, 73.379, 76.652]
phantom_brisque_stds = [3.851, 3.951, 4.127, 4.081, 4.169]

# --- Data for V_PHANTOM dataset ---
v_phantom_brisque_means = [77.634, 76.071, 75.678, 75.234, 72.931]
v_phantom_brisque_stds = [3.498, 3.359, 3.215, 3.246, 3.050]

# --- Plot BRISQUE scores ---
plot_brisque(mice_configs, mice_brisque_means, mice_brisque_stds, "MICE Dataset: Sparse 256 to Sparse 4")
plot_brisque(phantom_configs, phantom_brisque_means, phantom_brisque_stds, "Phantom Dataset: Sparse 128 to Sparse 8")
plot_brisque(phantom_configs, v_phantom_brisque_means, v_phantom_brisque_stds, "V Phantom Dataset: Sparse 128 to Sparse 8")