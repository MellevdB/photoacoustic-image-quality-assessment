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
    plt.xticks(x, configs, rotation=45, fontsize=14)
    plt.title(title, fontsize=20)
    plt.grid(True)
    plt.legend(fontsize=16)
    plt.show()

# Configurations
configs = ["PA2", "PA3", "PA4", "PA5", "PA6", "PA7"]

# Data for PA Experiment BRISQUE scores
pa_experiment_brisque = {
    "KneeSlice": ([53.600, 52.088, 49.637, 48.031, 47.224, 47.286], [1.623, 1.366, 0.953, 1.346, 1.470, 1.494]),
    "Phantoms": ([51.728, 48.584, 46.920, 46.400, 46.998, 48.258], [3.556, 3.111, 2.934, 2.844, 3.343, 3.667]),
    "SmallAnimal": ([55.061, 49.903, 47.373, 41.969, 40.159, 46.050], [1.498, 1.590, 1.350, 1.326, 1.974, 2.939]),
    "Transducers": ([42.092, 39.737, 39.787, 42.323, 50.655, 50.937], [3.348, 2.652, 2.197, 3.944, 6.591, 1.386])
}

# Generate BRISQUE plots
for dataset, (means, stds) in pa_experiment_brisque.items():
    plot_brisque(configs, means, stds, f"{dataset} Dataset")
