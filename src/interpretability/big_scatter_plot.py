import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy.stats import spearmanr, pearsonr

def create_scatter_grid(metric):
    """
    Create scatter plot grid for a given metric (S3IM or SSIM)
    """
    # Define models and dataset mapping
    models = ["best_model", "IQDCNN", "EfficientNetIQA"]
    datasets = {
        "A": ("mice",),
        "B": ("SWFD_sc",),
        "C": ("SCD_ms_ss32", "SCD_ms_ss64", "SCD_ms_ss128"),
        "D": ("SCD_vc_ms",),
        "E": ("pa_experiment_data",),
    }

    # === Figure settings ===
    fig = plt.figure(figsize=(18, 28))
    gs = gridspec.GridSpec(7, 3, height_ratios=[0.15, 1, 1, 1, 1, 1, 0.15],  # Added row for model names at top
                           wspace=0.05, hspace=0.05)

    row_keys = list(datasets.keys())

    # Set much larger font sizes
    tick_fontsize = 18
    label_fontsize = 28
    subplot_label_fontsize = 28
    model_label_fontsize = 32

    # Add model names at the top first
    for col_idx, model in enumerate(models):
        ax = fig.add_subplot(gs[0, col_idx])
        ax.axis("off")
        model_display = {
            "best_model": "PAQNet",
            "IQDCNN": "IQDCNN",
            "EfficientNetIQA": "EfficientNetIQA"
        }[model]
        ax.text(0.5, 0.5, model_display, ha="center", va="center",
                fontsize=model_label_fontsize, fontweight="bold", transform=ax.transAxes)

    # Loop over models and dataset groups
    for col_idx, model in enumerate(models):
        for row_idx, label in enumerate(row_keys):
            ax = fig.add_subplot(gs[row_idx + 1, col_idx])  # +1 because model names are now at index 0
            base_dir = "results/eval_model"
            df_list = []

            # Combine CSVs for each dataset group
            for ds in datasets[label]:
                path = os.path.join(base_dir, model, metric, ds, f"preds_vs_targets_{metric}_{ds}.csv")
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    df_list.append(df)

            if not df_list:
                print(f"Missing data for {model} - {label}")
                continue

            df_combined = pd.concat(df_list, ignore_index=True)
            ax.scatter(df_combined["target"], df_combined["prediction"],
                       alpha=0.6, edgecolor="k", s=20)

            # Axis limits and ticks
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ticks = np.arange(0.0, 1.1, 0.2)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.grid(True, linestyle='--', linewidth=0.5)
            ax.set_aspect('equal', adjustable='box')

            # Y-axis
            if col_idx > 0:
                ax.set_yticklabels([])
            else:
                ax.set_ylabel(f"Predicted {metric}", fontsize=label_fontsize)
                ax.tick_params(axis='y', labelsize=tick_fontsize)

            # X-axis
            if row_idx < len(row_keys) - 1:
                ax.set_xticklabels([])
            else:
                ax.set_xlabel(f"True {metric}", fontsize=label_fontsize, labelpad=16)
                ax.tick_params(axis='x', labelsize=tick_fontsize)

            # Add subplot label like A1, A2, etc.
            subplot_label = f"{label}{col_idx+1}"
            ax.text(0.02, 0.95, subplot_label,
                    transform=ax.transAxes, fontsize=subplot_label_fontsize,
                    fontweight='bold', va='top', ha='left')

    # Save figure
    output_path = f"combined_scatter_grid_{metric.lower()}_final_layout.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    
    return output_path

def create_combined_scatter_grid():
    """
    Create combined scatter plot grid with S3IM and SSIM side-by-side (6 columns total).
    Columns 0-2: S3IM [PAQNet, IQDCNN, EfficientNetIQA]
    Columns 3-5: SSIM [PAQNet, IQDCNN, EfficientNetIQA]
    Rows A-E. Subplots labeled A1..A6, B1..B6, etc.
    """
    # Define models and dataset mapping
    models = ["best_model", "IQDCNN", "EfficientNetIQA"]
    datasets = {
        "A": ("mice",),
        "B": ("SWFD_sc",),
        "C": ("SCD_ms_ss32", "SCD_ms_ss64", "SCD_ms_ss128"),
        "D": ("SCD_vc_ms",),
        "E": ("pa_experiment_data",),
    }
    metrics = ["S3IM", "SSIM"]

    # === Figure settings ===
    num_rows = len(datasets)
    num_cols = 6  # 3 models per metric, 2 metrics
    fig = plt.figure(figsize=(24, 24))  # Further reduced height from 28 to 24
    gs = gridspec.GridSpec(num_rows + 2, num_cols,  # +1 for top titles, +1 for bottom metric labels
                           height_ratios=[0.05] + [1] * num_rows + [0.08],  # Further reduced title and bottom ratios
                           wspace=0.4, hspace=0.02)

    row_keys = list(datasets.keys())

    # Font sizes
    tick_fontsize = 16
    axis_label_fontsize = 24
    subplot_label_fontsize = 22
    model_label_fontsize = 24
    metric_label_fontsize = 24

    # Column titles (model names repeated for S3IM and SSIM halves)
    for col_idx in range(num_cols):
        model_key = models[col_idx % 3]
        ax = fig.add_subplot(gs[0, col_idx])
        ax.axis("off")
        model_display = {
            "best_model": "PAQNet",
            "IQDCNN": "IQDCNN",
            "EfficientNetIQA": "EfficientNetIQA"
        }[model_key]
        ax.text(0.5, 0.8, model_display, ha="center", va="bottom",
                fontsize=model_label_fontsize, fontweight="bold", transform=ax.transAxes)

    # Main grid of subplots
    for row_idx, label in enumerate(row_keys):
        for col_idx in range(num_cols):
            # Determine metric and model per column
            metric = metrics[0] if col_idx < 3 else metrics[1]
            model_key = models[col_idx % 3]

            ax = fig.add_subplot(gs[row_idx + 1, col_idx])
            base_dir = "results/eval_model"
            df_list = []

            # Combine CSVs for each dataset group
            for ds in datasets[label]:
                path = os.path.join(base_dir, model_key, metric, ds, f"preds_vs_targets_{metric}_{ds}.csv")
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    df_list.append(df)

            if not df_list:
                print(f"Missing data for {model_key} - {label} - {metric}")
                continue

            df_combined = pd.concat(df_list, ignore_index=True)
            ax.scatter(df_combined["target"], df_combined["prediction"],
                       alpha=0.6, edgecolor="k", s=20)

            # Axes and ticks
            ax.set_xlim(0.0, 1.0)
            ax.set_ylim(0.0, 1.0)
            ticks = np.arange(0.0, 1.1, 0.2)
            ax.set_xticks(ticks)
            ax.set_yticks(ticks)
            ax.grid(True, linestyle='--', linewidth=0.5)
            ax.set_aspect('equal', adjustable='box')

            # Y tick labels on all subplots
            ax.tick_params(axis='y', labelsize=tick_fontsize)
            # Only first column gets Y-axis title
            if col_idx == 0:
                ax.set_ylabel("Predicted Score", fontsize=axis_label_fontsize)

            # X-axis: show numeric tick labels only for bottom row; keep title only on last row
            if row_idx == num_rows - 1:
                ax.set_xlabel("True Score", fontsize=axis_label_fontsize, labelpad=8)
                ax.tick_params(axis='x', labelsize=tick_fontsize)
            else:
                ax.set_xticklabels([])
                ax.tick_params(axis='x', which='both', length=3)

            # Subplot label A1..A6, B1..B6, etc.
            subplot_number = col_idx + 1
            subplot_label = f"{label}{subplot_number}"
            ax.text(0.02, 0.95, subplot_label,
                    transform=ax.transAxes, fontsize=subplot_label_fontsize,
                    fontweight='bold', va='top', ha='left')

    # Bottom metric labels centered under left/right halves
    # Left half: S3IM under columns 0-2
    ax_left = fig.add_subplot(gs[num_rows + 1, 1])
    ax_left.axis("off")
    ax_left.text(0.5, 0.2, "S3IM", ha="center", va="center",
                 fontsize=metric_label_fontsize, fontweight="bold", transform=ax_left.transAxes)

    # Right half: SSIM under columns 3-5
    ax_right = fig.add_subplot(gs[num_rows + 1, 4])
    ax_right.axis("off")
    ax_right.text(0.5, 0.2, "SSIM", ha="center", va="center",
                  fontsize=metric_label_fontsize, fontweight="bold", transform=ax_right.transAxes)

    # No need for subplots_adjust - GridSpec handles spacing

    # Save figure
    output_path = "combined_scatter_grid_s3im_ssim_final_layout.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    return output_path

def compute_combined_correlations(metric):
    """
    Compute combined correlation for SCD_ms across all models for a given metric
    """
    scd_ms_subsets = ["SCD_ms_ss32", "SCD_ms_ss64", "SCD_ms_ss128"]
    models_dict = {
        "best_model": "PAQNet",
        "IQDCNN": "IQDCNN",
        "EfficientNetIQA": "EfficientNetIQA"
    }

    print(f"\n=== Combined SCD_ms correlations for {metric} ===")
    for model_key, model_name in models_dict.items():
        combined_df_list = []
        base_dir = os.path.join("results", "eval_model", model_key, metric)

        for subset in scd_ms_subsets:
            csv_path = os.path.join(base_dir, subset, f"preds_vs_targets_{metric}_{subset}.csv")
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                combined_df_list.append(df)
            else:
                print(f"Missing file for {model_name}: {csv_path}")

        if combined_df_list:
            df_combined = pd.concat(combined_df_list, ignore_index=True)
            spearman_corr, _ = spearmanr(df_combined["target"], df_combined["prediction"])
            pearson_corr, _ = pearsonr(df_combined["target"], df_combined["prediction"])
            print(f"{model_name}:")
            print(f"  Spearman: {spearman_corr:.4f}")
            print(f"  Pearson:  {pearson_corr:.4f}\n")
        else:
            print(f"No SCD_ms data found for {model_name}\n")

# Generate plots for both metrics
print("Generating S3IM scatter plot...")
s3im_output = create_scatter_grid("S3IM")
print(f"S3IM plot saved as: {s3im_output}")

print("Generating SSIM scatter plot...")
ssim_output = create_scatter_grid("SSIM")
print(f"SSIM plot saved as: {ssim_output}")

print("Generating combined S3IM+SSIM scatter plot...")
combined_output = create_combined_scatter_grid()
print(f"Combined plot saved as: {combined_output}")

# Compute correlations for both metrics
compute_combined_correlations("S3IM")
compute_combined_correlations("SSIM")