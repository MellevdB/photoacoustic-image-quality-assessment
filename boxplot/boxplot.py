import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind

# === PATH SETTINGS ===
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
OUTPUT_DIR = os.path.dirname(__file__)  # save PNGs next to this script

# === PER-IMAGE CSV PATHS ===
CSV_PATHS = {
    'denoising':      os.path.join(RESULTS_DIR, "denoising_data/denoising_data_per_image_metrics_2025-05-20_13-29-51.csv"),
    'mice':           os.path.join(RESULTS_DIR, "mice/mice_per_image_metrics_2025-05-20_13-29-51.csv"),
    'msfd':           os.path.join(RESULTS_DIR, "MSFD/MSFD_per_image_metrics_2025-05-20_13-29-51.csv"),
    'pa_experiment':  os.path.join(RESULTS_DIR, "pa_experiment_data/pa_experiment_data_per_image_metrics_2025-05-20_13-29-51.csv"),
    'phantom':        os.path.join(RESULTS_DIR, "phantom/phantom_per_image_metrics_2025-05-20_13-29-51.csv"),
    'scd':            os.path.join(RESULTS_DIR, "SCD/SCD_per_image_metrics_2025-05-20_13-29-51.csv"),
    'swfd':           os.path.join(RESULTS_DIR, "SWFD/SWFD_per_image_metrics_2025-05-20_13-29-51.csv"),
    'v_phantom':      os.path.join(RESULTS_DIR, "v_phantom/v_phantom_per_image_metrics_2025-05-20_13-29-51.csv"),
}

# === MANUAL CONFIG ORDER ===
manual_config_sort = {
    "denoising": ['10db', '20db', '30db', '40db', '50db'],
    "mice": ['sparse4_recon_all', 'sparse8_recon_all', 'sparse16_recon_all',
             'sparse32_recon_all', 'sparse64_recon_all', 'sparse128_recon_all', 'sparse256_recon_all'],
    "msfd": ['ms,ss32_BP_w760', 'ms,ss64_BP_w760', 'ms,ss128_BP_w760'],
    "scd_ms": ['ms,ss32_BP', 'ms,ss64_BP', 'ms,ss128_BP'],
    "scd_vc": ['vc,ss32_BP', 'vc,ss64_BP', 'vc,ss128_BP'],
    "swfd_ms": ['ms,ss32_BP', 'ms,ss64_BP', 'ms,ss128_BP'],
    "swfd_sc": ['sc,ss32_BP', 'sc,ss64_BP', 'sc,ss128_BP'],
    "phantom": ['BP_phantom_8', 'BP_phantom_16', 'BP_phantom_32', 'BP_phantom_64', 'BP_phantom_128'],
    "v_phantom": ['v_phantom_8', 'v_phantom_16', 'v_phantom_32', 'v_phantom_64', 'v_phantom_128'],
}

# === LOAD & COMBINE DATA ===
all_data = []
for dataset_key, csv_path in CSV_PATHS.items():
    if not os.path.exists(csv_path):
        print(f"[SKIP] Missing: {csv_path}")
        continue
    df = pd.read_csv(csv_path)
    df["Configuration"] = df["configuration"].astype(str).str.strip()

    if dataset_key == "msfd":
        df = df[df["Configuration"].str.contains("_w760")]
        df["Dataset"] = "msfd"
    elif dataset_key == "scd":
        df_ms = df[df["Configuration"].str.startswith("ms,")].copy()
        df_vc = df[df["Configuration"].str.startswith("vc,") & ~df["Configuration"].str.contains("lv128")].copy()
        df_ms["Dataset"] = "scd_ms"
        df_vc["Dataset"] = "scd_vc"
        all_data.extend([df_ms, df_vc])
        continue
    elif dataset_key == "swfd":
        df_ms = df[df["Configuration"].str.startswith("ms,")].copy()
        df_sc = df[df["Configuration"].str.startswith("sc,") & ~df["Configuration"].str.contains("lv128")].copy()
        df_ms["Dataset"] = "swfd_ms"
        df_sc["Dataset"] = "swfd_sc"
        all_data.extend([df_ms, df_sc])
        continue
    else:
        df["Dataset"] = dataset_key

    all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)

META_COLS = ["dataset", "configuration", "ground_truth", "wavelength", "image_path", "Dataset", "Configuration"]
all_metrics = [c for c in df_all.columns if c not in META_COLS]

# === ASSIGN "Subset 1..n" PER DATASET ===
df_all["Subset"] = None
for dataset in df_all["Dataset"].unique():
    configs = manual_config_sort.get(dataset, sorted(df_all[df_all["Dataset"] == dataset]["Configuration"].unique()))
    for i, cfg in enumerate(configs):
        label = f"Subset {i+1}"
        df_all.loc[(df_all["Dataset"] == dataset) & (df_all["Configuration"] == cfg), "Subset"] = label

# === SIGNIFICANCE (adjacent pairs only, same as before) ===
ttest_results = []
for metric in all_metrics:
    for dataset in df_all["Dataset"].unique():
        subset = df_all[df_all["Dataset"] == dataset].dropna(subset=[metric])
        configs = manual_config_sort.get(dataset, sorted(subset["Configuration"].unique()))
        for i in range(len(configs) - 1):
            c1, c2 = configs[i], configs[i + 1]
            v1 = subset[subset["Configuration"] == c1][metric]
            v2 = subset[subset["Configuration"] == c2][metric]
            if len(v1) < 2 or len(v2) < 2:
                continue
            _, p = ttest_ind(v1, v2, equal_var=False)
            ttest_results.append({"Metric": metric, "Dataset": dataset, "Config1": c1, "Config2": c2, "P-Value": p})

sig_df = pd.DataFrame(ttest_results)

def p_to_star(p):
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return None

def add_significance(ax, metric, dataset, data, sig_df, offset=0.02):
    sub = data[data["Dataset"] == dataset]
    order = sorted(sub["Subset"].unique())
    if len(order) < 2:
        return

    # map Subset -> config
    map_rows = sub[["Subset", "Configuration"]].drop_duplicates()
    s2c = dict(zip(map_rows["Subset"], map_rows["Configuration"]))

    weakest = "***"
    for i in range(len(order) - 1):
        c1, c2 = s2c[order[i]], s2c[order[i + 1]]
        row = sig_df[(sig_df["Metric"] == metric) &
                     (sig_df["Dataset"] == dataset) &
                     (((sig_df["Config1"] == c1) & (sig_df["Config2"] == c2)) |
                      ((sig_df["Config1"] == c2) & (sig_df["Config2"] == c1)))]
        if row.empty:
            return
        star = p_to_star(row["P-Value"].values[0])
        if star is None:
            return
        if len(star) < len(weakest):
            weakest = star

    # draw just the star above the current axis range
    y = sub[metric].max()
    ax.text(0.5, y + offset, weakest, ha="center", va="bottom",
            transform=ax.get_xaxis_transform(), fontsize=14)

# === PLOT (two pages) ===
filtered_metrics = [m for m in all_metrics if "_norm" not in m and m != "TV"][:13]
chunks = [filtered_metrics[:7], filtered_metrics[7:13]]

for page_idx, chunk in enumerate(chunks, start=1):
    rows = len(chunk)
    # dynamic, safe size per row
    fig_w = 18
    fig_h = max(3.8 * rows + 1.5, 12)  # ~3.8 in/row keeps text legible
    fig, axes = plt.subplots(
        nrows=rows, ncols=1, figsize=(fig_w, fig_h),
        sharex=False, constrained_layout=True
    )

    if rows == 1:
        axes = [axes]

    for i, metric in enumerate(chunk):
        ax = axes[i]
        plot_data = df_all.dropna(subset=[metric]).copy()
        plot_data["Dataset"] = plot_data["Dataset"].replace({"denoising": "NNE", "pa_experiment": "EFA"})

        sns.boxplot(x="Dataset", y=metric, hue="Subset",
                    data=plot_data, palette="Set2", ax=ax,
                    showfliers=True, width=0.7)

        # panel label
        panel_letter = chr(65 + (i + (page_idx - 1) * 7))
        ax.text(0.98, 0.93, panel_letter, transform=ax.transAxes,
                ha="right", va="top", fontsize=15,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))

        ax.set_ylabel(metric, fontsize=16)
        ax.get_legend().remove()

        # significance stars per dataset group
        for ds in plot_data["Dataset"].unique():
            add_significance(ax, metric, ds, plot_data, sig_df)

        # rotate x labels only on the bottom subplot
        if i == rows - 1:
            ax.tick_params(axis='x', labelrotation=45, labelsize=13)
        else:
            ax.tick_params(axis='x', labelbottom=False)

        ax.tick_params(axis='y', labelsize=12)

    # shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    def sort_key(lbl):
        m = re.search(r"(\d+)", lbl)
        return int(m.group(1)) if m else 999
    pairs = sorted(zip(handles, labels), key=lambda x: sort_key(x[1]))
    handles, labels = zip(*pairs) if pairs else ([], [])

    fig.legend(handles, labels, title="Subset",
               loc='lower center', ncol=min(8, len(labels)) if labels else 1,
               fontsize=12, title_fontsize=13, bbox_to_anchor=(0.5, 0.0))

    out_path = os.path.join(OUTPUT_DIR, f"combined_metrics_boxplot_final_page{page_idx}.png")
    # No bbox_inches="tight" to avoid bbox expansion; DPI 200 keeps detail without huge files
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[INFO] Saved: {out_path}")