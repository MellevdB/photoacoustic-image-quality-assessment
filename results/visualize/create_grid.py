import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from matplotlib.gridspec import GridSpec

# Plot directory and datasets
plot_dir = "results/visualize/plots"

grouped_datasets = [
    [None, "denoising", "pa_experiment_KneeSlice1", None],  # row 1 → 2 plots centered
    ["pa_experiment_Phantoms", "pa_experiment_Transducers", "pa_experiment_SmallAnimal"],
    ["mice", "phantom", "v_phantom"],
    ["scd_virtual_circle", "msfd", "scd_multi_segment"],
    [None, "swfd_multi_segment", "swfd_semi_circle", None]   # row 5 → 2 plots centered
]

def create_a4_layout(metric_type, save_name):
    # Define A4 size: width x height in inches
    fig_width = 8.27
    fig_height = 11.69
    fig = plt.figure(figsize=(fig_width, fig_height))

    n_rows = len(grouped_datasets)
    height_ratios = [1] * n_rows  # All rows same height
    gs = GridSpec(nrows=n_rows, ncols=4, figure=fig, height_ratios=height_ratios, wspace=0.05, hspace=0.05)

    for row_idx, row in enumerate(grouped_datasets):
        col_positions = [0, 1, 2, 3]
        n_items = len(row)

        # Determine how many actual plots and adjust positions accordingly
        plot_indices = [i for i, val in enumerate(row) if val is not None]
        for i, dataset in enumerate(row):
            ax = fig.add_subplot(gs[row_idx, col_positions[i]])
            if dataset is None:
                ax.axis('off')
                continue

            file_path = os.path.join(plot_dir, f"{dataset}_{metric_type}.png")
            if os.path.exists(file_path):
                img = mpimg.imread(file_path)
                ax.imshow(img)
                ax.set_title(dataset.replace("pa_experiment_", "").replace("_", " ").title(), fontsize=8)
            else:
                print(f"[WARNING] File not found: {file_path}")
            ax.axis('off')

    plt.tight_layout(pad=0.5)
    plt.savefig(os.path.join(plot_dir, save_name), dpi=300)
    plt.close()

# Generate both figures
create_a4_layout("FR", "all_datasets_FR_A4grid.png")
create_a4_layout("NR", "all_datasets_NR_A4grid.png")