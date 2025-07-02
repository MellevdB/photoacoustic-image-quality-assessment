import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# Define models and dataset mapping (5 datasets total)
models = ["best_model", "IQDCNN", "EfficientNetIQA"]
datasets = {
    "A": ("mice",),
    "B": ("SWFD_sc",),
    "C": ("SCD_ms_ss32", "SCD_ms_ss64", "SCD_ms_ss128"),
    "D": ("SCD_vc_ms",),
    "E": ("pa_experiment_data",),
}
metric = "S3IM"

# Start plotting
fig = plt.figure(figsize=(18, 28))  # taller for label + model name
gs = gridspec.GridSpec(6, 3, height_ratios=[1, 1, 1, 1, 1, 0.15], wspace=0.05, hspace=0.15)
row_keys = list(datasets.keys())

for col_idx, model in enumerate(models):
    for row_idx, label in enumerate(row_keys):
        ax = fig.add_subplot(gs[row_idx, col_idx])
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
        ax.scatter(df_combined["target"], df_combined["prediction"], alpha=0.6, edgecolor="k", s=20)

        # Axis limits and ticks every 0.1
        ax.set_xlim(0.0, 1.0)
        ax.set_ylim(0.0, 1.0)
        ticks = np.arange(0.0, 1.1, 0.2)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.grid(True, linestyle='--', linewidth=0.5)
        ax.set_aspect('equal', adjustable='box')

        # Y-axis settings
        if col_idx > 0:
            ax.set_yticklabels([])
        else:
            ax.set_ylabel("Predicted S3IM", fontsize=20)
            ax.tick_params(axis='y', labelsize=16)

        # X-axis settings
        if row_idx < 4:
            ax.set_xticklabels([])
        else:
            ax.set_xlabel("True S3IM", fontsize=20, labelpad=10)
            ax.tick_params(axis='x', labelsize=16)

        # Add row label (A–E)
        if col_idx == 0:
            ax.text(-0.25, 1.05, label, transform=ax.transAxes,
                    fontsize=24, fontweight='bold', va='top', ha='right')

# Add model labels under each column (extra vertical space with y=-0.5)
for col_idx, model in enumerate(models):
    ax = fig.add_subplot(gs[5, col_idx])
    ax.axis("off")
    model_display = {
        "best_model": "PhotoacousticQualityNet",
        "IQDCNN": "IQDCNN",
        "EfficientNetIQA": "EfficientNetIQA"
    }[model]
    ax.text(0.5, -0.5, model_display, ha="center", va="top", fontsize=22, fontweight="bold", transform=ax.transAxes)

# Save final figure
output_path = "combined_scatter_grid_s3im_final_layout.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.close()

from scipy.stats import spearmanr, pearsonr

# === Compute combined correlation for SCD_ms across all models ===
scd_ms_subsets = ["SCD_ms_ss32", "SCD_ms_ss64", "SCD_ms_ss128"]
models = {
    "best_model": "PhotoacousticQualityNet",
    "IQDCNN": "IQDCNN",
    "EfficientNetIQA": "EfficientNetIQA"
}
metric = "S3IM"

print("\n=== Combined SCD_ms correlations ===")
for model_key, model_name in models.items():
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