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

# Data for PA Experiment BRISQUE scores with mean normalization + bandpass filtering
pa_experiment_brisque_mean = {
    "KneeSlice": ([53.600, 52.088, 49.637, 48.031, 47.224, 47.286], [1.623, 1.366, 0.953, 1.346, 1.470, 1.494]),
    "Phantoms": ([51.728, 48.584, 46.920, 46.400, 46.998, 48.258], [3.556, 3.111, 2.934, 2.844, 3.343, 3.667]),
    "SmallAnimal": ([55.061, 49.903, 47.373, 41.969, 40.159, 46.050], [1.498, 1.590, 1.350, 1.326, 1.974, 2.939]),
    "Transducers": ([42.092, 39.737, 39.787, 42.323, 50.655, 50.937], [3.348, 2.652, 2.197, 3.944, 6.591, 1.386])
}

# Data for PA Experiment BRISQUE scores with zscore normalization + bandpass filtering
pa_experiment_brisque_zscore = {
    "KneeSlice": ([53.503, 51.681, 49.496, 48.117, 47.709, 47.246], [1.771, 1.437, 0.874, 0.856, 1.307, 1.467]),
    "Phantoms": ([50.485, 47.299, 45.574, 44.954, 45.660, 48.528], [3.452, 3.073, 3.301, 2.885, 4.152, 5.634]),
    "SmallAnimal": ([52.937, 47.779, 44.403, 38.869, 37.031, 43.748], [2.046, 1.469, 1.361, 1.157, 1.325, 2.872]),
    "Transducers": ([42.984, 40.558, 39.636, 42.773, 51.885, 50.948], [3.747, 2.706, 3.133, 2.740, 2.509, 0.305])
}

# Results no additional preprocessing
pa_experiment_brisque_no_preprocessing = {
    "KneeSlice1": ([168.769, 162.175, 165.365, 153.699, 154.712, 155.846], [8.230, 6.791, 7.946, 8.956, 5.240, 0.006]),
    "Phantoms": ([159.095, 157.025, 156.089, 155.819, 155.705, 155.570], [9.169, 7.350, 3.505, 6.152, 0.812, 2.254]),
    "SmallAnimal": ([158.795, 153.037, 154.095, 155.845, 155.845, 155.845], [8.138, 9.748, 7.498, 0.000, 0.000, 0.000]),
    "Transducers": ([163.694, 157.110, 158.481, 154.723, 155.845, 155.845], [9.462, 10.661, 4.567, 1.943, 0.000, 0.000])
}


# Generate BRISQUE plots
# for dataset, (means, stds) in pa_experiment_brisque_mean.items():
#     plot_brisque(configs, means, stds, f"{dataset} Dataset")


# for dataset, (means, stds) in pa_experiment_brisque_zscore.items():
#     plot_brisque(configs, means, stds, f"{dataset} Dataset")

for dataset, (means, stds) in pa_experiment_brisque_no_preprocessing.items():
    plot_brisque(configs, means, stds, f"{dataset} Dataset")
