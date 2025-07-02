import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import shlex

# === CONFIGURATIONS ===

DENOISING_CONFIGS = ['10db', '20db', '30db', '40db', '50db']
MICE_CONFIGS = ['sparse4_recon_all', 'sparse8_recon_all', 'sparse16_recon_all',
                'sparse32_recon_all', 'sparse64_recon_all', 'sparse128_recon_all', 'sparse256_recon_all']
PHANTOM_CONFIGS = ['BP_phantom_8', 'BP_phantom_16', 'BP_phantom_32', 'BP_phantom_64', 'BP_phantom_128']
V_PHANTOM_CONFIGS = ['v_phantom_8', 'v_phantom_16', 'v_phantom_32', 'v_phantom_64', 'v_phantom_128']
MSFD_CONFIGS = ['ms,ss32_BP_w760', 'ms,ss64_BP_w760', 'ms,ss128_BP_w760']
SCD_CONFIGS = {
    'virtual_circle': ['vc,ss32_BP', 'vc,ss64_BP', 'vc,ss128_BP'],
    'multi_segment': ['ms,ss32_BP', 'ms,ss64_BP', 'ms,ss128_BP']
}
SWFD_CONFIGS = {
    'multi_segment': ['ms,ss32_BP', 'ms,ss64_BP', 'ms,ss128_BP'],
    'semi_circle': ['sc,ss32_BP', 'sc,ss64_BP', 'sc,ss128_BP']
}
PA_EXPERIMENT_CONFIGS = {
    'KneeSlice1': ['PA7', 'PA6', 'PA5', 'PA4', 'PA3', 'PA2'],
    'Phantoms': ['PA7', 'PA6', 'PA5', 'PA4', 'PA3', 'PA2'],
    'Transducers': ['PA7', 'PA6', 'PA5', 'PA4', 'PA3', 'PA2'],
    'SmallAnimal': ['PA7', 'PA6', 'PA5', 'PA4', 'PA3', 'PA2']
}

DATASET_CONFIGS = {
    'denoising': DENOISING_CONFIGS,
    'mice': MICE_CONFIGS,
    'phantom': PHANTOM_CONFIGS,
    'v_phantom': V_PHANTOM_CONFIGS,
    'msfd': MSFD_CONFIGS,
    'scd': SCD_CONFIGS,
    'swfd': SWFD_CONFIGS,
    'pa_experiment': PA_EXPERIMENT_CONFIGS
}

# === CONFIGURATION ORDER MAPPING ===
manual_config_sort = {
    "denoising": ['10db', '20db', '30db', '40db', '50db'],
    "mice": ['sparse4_recon_all', 'sparse8_recon_all', 'sparse16_recon_all', 'sparse32_recon_all', 'sparse64_recon_all', 'sparse128_recon_all', 'sparse256_recon_all'],
    "phantom": ['BP_phantom_8', 'BP_phantom_16', 'BP_phantom_32', 'BP_phantom_64', 'BP_phantom_128'],
    "v_phantom": ['v_phantom_8', 'v_phantom_16', 'v_phantom_32', 'v_phantom_64', 'v_phantom_128'],
    "msfd": ['ms,ss32_BP_w760', 'ms,ss64_BP_w760', 'ms,ss128_BP_w760'],
    "scd_ms": ['ms,ss32_BP', 'ms,ss64_BP', 'ms,ss128_BP'],
    "scd_vc": ['vc,ss32_BP', 'vc,ss64_BP', 'vc,ss128_BP', 'vc,lv128_BP'],
    "swfd_ms": ['ms,ss32_BP', 'ms,ss64_BP', 'ms,ss128_BP'],
    "swfd_sc": ['sc,ss32_BP', 'sc,ss64_BP', 'sc,ss128_BP', 'sc,lv128_BP'],
    "pa_experiment": ['PA2', 'PA3', 'PA4', 'PA5', 'PA6', 'PA7']  # PA1 is ground truth, not a config
}

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

DATASET_results_paths = {
    'denoising': os.path.join(PROJECT_ROOT, "results/denoising_data/denoising_data_results_2025-05-20_13-29-51.txt"),
    'mice': os.path.join(PROJECT_ROOT, "results/mice/mice_results_2025-05-20_13-29-51.txt"),
    'phantom': os.path.join(PROJECT_ROOT, "results/phantom/phantom_results_2025-05-20_13-29-51.txt"),
    'v_phantom': os.path.join(PROJECT_ROOT, "results/v_phantom/v_phantom_results_2025-05-20_13-29-51.txt"),
    'msfd': os.path.join(PROJECT_ROOT, "results/MSFD/MSFD_results_2025-05-20_13-29-51.txt"),
    'scd': os.path.join(PROJECT_ROOT, "results/SCD/SCD_results_2025-05-20_13-29-51.txt"),
    'swfd': os.path.join(PROJECT_ROOT, "results/SWFD/SWFD_results_2025-05-20_13-29-51.txt"),
    'pa_experiment': os.path.join(PROJECT_ROOT, "results/pa_experiment_data/pa_experiment_data_results_2025-05-20_13-29-51.txt")
}

FR_METRICS = ["PSNR", "SSIM", "MSSSIM", "IWSSIM", "VIF", "FSIM", "GMSD", "MSGMSD", "HAARPSI", "UQI", "S3IM"]
GMSD_METRICS = ["GMSD", "MSGMSD"]
FR_TOP = ["PSNR"]
FR_BOTTOM = [m for m in FR_METRICS if m not in FR_TOP + GMSD_METRICS]
NR_TOP = ["BRISQUE", "TV"]
NR_BOTTOM = ["CLIP-IQA"]

colors = {
    "FSIM": "b", "UQI": "g", "PSNR": "r", "SSIM": "c", "VIF": "m", "S3IM": "y",
    "MSSSIM": "tab:blue", "IWSSIM": "tab:orange", "GMSD": "tab:green",
    "MSGMSD": "tab:red", "HAARPSI": "tab:purple",
    "BRISQUE": "gray", "TV": "darkgreen", "CLIP-IQA": "gold"
}

def load_dataset_results(result_file):
    with open(result_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    lines = [line for line in lines if not re.fullmatch(r"-+", line)]
    raw_header = shlex.split(lines[0])
    fixed_header = []
    i = 0
    while i < len(raw_header):
        if raw_header[i] == "Ground" and i + 1 < len(raw_header) and raw_header[i + 1] == "Truth":
            fixed_header.append("Ground Truth")
            i += 2
        else:
            fixed_header.append(raw_header[i])
            i += 1
    data_rows = []
    for i, line in enumerate(lines[1:]):
        row = shlex.split(line)
        if len(row) != len(fixed_header):
            print(f"[WARNING] Row {i} has {len(row)} columns but expected {len(fixed_header)}: {row}")
        data_rows.append(row)
    df = pd.DataFrame(data_rows, columns=fixed_header)
    for col in df.columns:
        try: df[col] = pd.to_numeric(df[col])
        except: pass
    return df

def extract_metrics(df, metrics, config_list):
    config_col = "Configuration"
    df = df.copy()
    df[config_col] = df[config_col].astype(str).str.strip() 
    config_list = [cfg.strip() for cfg in config_list]
    filtered_df = df[df[config_col].isin(config_list)]
    filtered_df = filtered_df.set_index(config_col).loc[config_list].reset_index()
    metric_group = {}
    for metric in metrics:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col in filtered_df.columns:
            try:
                means = pd.to_numeric(filtered_df[mean_col], errors="raise").tolist()
                stds = pd.to_numeric(filtered_df[std_col], errors="raise").tolist() if std_col in filtered_df.columns else [0]*len(filtered_df)
                metric_group[metric] = {"mean": means, "std": stds}
            except Exception as e:
                print(f"[ERROR] Failed to process metric '{metric}':", e)
    return metric_group

def plot_gmsd(configs, metric_group, title, filename):
    x = np.arange(len(configs))
    plt.figure(figsize=(10, 5))
    for metric, values in metric_group.items():
        means = np.array(values['mean'])
        stds = np.array(values['std'])
        color = colors.get(metric, 'k')
        plt.plot(x, means, label=metric, color=color, marker="o")
        plt.fill_between(x, means - stds, means + stds, color=color, alpha=0.2)
    plt.xticks(x, configs, rotation=45, fontsize=14)
    plt.ylabel("Metric Value", fontsize=16)
    plt.ylim(0, 0.4)
    plt.title(title, fontsize=18)
    plt.legend()
    plt.tight_layout()
    out_dir = os.path.join(PROJECT_ROOT, "results/visualize/plots")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()

def plot_dual_metrics(configs, metric_groups, title, filename):
    x = np.arange(len(configs))
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(12, 8),
                                            gridspec_kw={'height_ratios': [1, 1]})
    for group, metrics, axis in [("top", metric_groups['top'], ax_top), ("bottom", metric_groups['bottom'], ax_bottom)]:
        for metric, values in metrics.items():
            means = np.array(values['mean'])
            stds = np.array(values['std'])
            color = colors.get(metric, 'k')
            axis.plot(x, means, label=metric, color=color, linestyle="--" if group == "top" else "-", marker="o")
            axis.fill_between(x, means - stds, means + stds, color=color, alpha=0.2)
        axis.set_ylabel("Metric Value", fontsize=16)
        axis.legend(loc="upper left", fontsize=12)

    ax_top.spines['bottom'].set_visible(False)
    ax_bottom.spines['top'].set_visible(False)
    ax_top.xaxis.tick_top()
    ax_top.tick_params(labeltop=False)
    ax_bottom.xaxis.tick_bottom()
    d = .015
    kwargs = dict(transform=ax_top.transAxes, color='k', clip_on=False)
    ax_top.plot((-d, +d), (-d, +d), **kwargs)
    ax_top.plot((1 - d, 1 + d), (-d, +d), **kwargs)
    kwargs.update(transform=ax_bottom.transAxes)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), **kwargs)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)
    ax_bottom.set_xticks(x)
    ax_bottom.set_xticklabels(configs, rotation=45, fontsize=14)
    plt.suptitle(title, fontsize=18)
    plt.tight_layout()
    out_dir = os.path.join(PROJECT_ROOT, "results/visualize/plots")
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, filename))
    plt.close()

# # === MAIN LOOP ===
# for dataset_key, result_path in DATASET_results_paths.items():
#     print("Dataset key:", dataset_key)
#     if not os.path.exists(result_path):
#         print(f"Skipping {dataset_key} (missing file)")
#         continue
#     df = load_dataset_results(result_path)
#     configs_entry = DATASET_CONFIGS[dataset_key]

#     if dataset_key == "pa_experiment":
#         df["Subgroup"] = df["Dataset"].apply(lambda x: x.split("/")[-1].strip())
#         for subgroup, config_list in configs_entry.items():
#             df_sub = df[df["Subgroup"] == subgroup].copy()
#             if df_sub.empty:
#                 print(f"[WARNING] No rows matched subgroup {subgroup}")
#                 continue
#             df_sub["Configuration"] = df_sub["Configuration"].astype(str).str.strip()

#             fr_top_metrics = extract_metrics(df_sub, FR_TOP, config_list)
#             fr_bot_metrics = extract_metrics(df_sub, FR_BOTTOM, config_list)
#             gmsd_metrics = extract_metrics(df_sub, GMSD_METRICS, config_list)
#             nr_top_metrics = extract_metrics(df_sub, NR_TOP, config_list)
#             nr_bot_metrics = extract_metrics(df_sub, NR_BOTTOM, config_list)

#             plot_dual_metrics(config_list, {"top": fr_top_metrics, "bottom": fr_bot_metrics},
#                               f"{dataset_key.upper()} - {subgroup} (FR)", f"{dataset_key}_{subgroup}_FR.png")
#             plot_dual_metrics(config_list, {"top": nr_top_metrics, "bottom": nr_bot_metrics},
#                               f"{dataset_key.upper()} - {subgroup} (NR)", f"{dataset_key}_{subgroup}_NR.png")
#             plot_gmsd(config_list, gmsd_metrics,
#                       f"{dataset_key.upper()} - {subgroup} (GMSD)", f"{dataset_key}_{subgroup}_GMSD.png")

#     elif isinstance(configs_entry, dict):
#         for subcat, config_list in configs_entry.items():
#             df_sub = df[df["Configuration"].isin(config_list)]
#             if df_sub.empty:
#                 print(f"[WARNING] No configs found for {dataset_key} - {subcat}")
#                 continue
#             fr_top_metrics = extract_metrics(df_sub, FR_TOP, config_list)
#             fr_bot_metrics = extract_metrics(df_sub, FR_BOTTOM, config_list)
#             gmsd_metrics = extract_metrics(df_sub, GMSD_METRICS, config_list)
#             nr_top_metrics = extract_metrics(df_sub, NR_TOP, config_list)
#             nr_bot_metrics = extract_metrics(df_sub, NR_BOTTOM, config_list)

#             plot_dual_metrics(config_list, {"top": fr_top_metrics, "bottom": fr_bot_metrics},
#                               f"{dataset_key.upper()} - {subcat} (FR)", f"{dataset_key}_{subcat}_FR.png")
#             plot_dual_metrics(config_list, {"top": nr_top_metrics, "bottom": nr_bot_metrics},
#                               f"{dataset_key.upper()} - {subcat} (NR)", f"{dataset_key}_{subcat}_NR.png")
#             plot_gmsd(config_list, gmsd_metrics,
#                       f"{dataset_key.upper()} - {subcat} (GMSD)", f"{dataset_key}_{subcat}_GMSD.png")

#     else:
#         config_list = configs_entry
#         fr_top_metrics = extract_metrics(df, FR_TOP, config_list)
#         fr_bot_metrics = extract_metrics(df, FR_BOTTOM, config_list)
#         gmsd_metrics = extract_metrics(df, GMSD_METRICS, config_list)
#         nr_top_metrics = extract_metrics(df, NR_TOP, config_list)
#         nr_bot_metrics = extract_metrics(df, NR_BOTTOM, config_list)

#         plot_dual_metrics(config_list, {"top": fr_top_metrics, "bottom": fr_bot_metrics},
#                           f"{dataset_key.upper()} (FR)", f"{dataset_key}_FR.png")
#         plot_dual_metrics(config_list, {"top": nr_top_metrics, "bottom": nr_bot_metrics},
#                           f"{dataset_key.upper()} (NR)", f"{dataset_key}_NR.png")
#         plot_gmsd(config_list, gmsd_metrics,
#                   f"{dataset_key.upper()} (GMSD)", f"{dataset_key}_GMSD.png")

import seaborn as sns
from scipy.stats import ttest_ind
from collections import defaultdict
from itertools import combinations

# Paths to per-image metric CSVs
PER_IMAGE_CSV_PATHS = {
    'denoising': 'results/denoising_data/denoising_data_per_image_metrics_2025-05-20_13-29-51.csv',
    'mice': 'results/mice/mice_per_image_metrics_2025-05-20_13-29-51.csv',
    'msfd': 'results/MSFD/MSFD_per_image_metrics_2025-05-20_13-29-51.csv',
    'pa_experiment': 'results/pa_experiment_data/pa_experiment_data_per_image_metrics_2025-05-20_13-29-51.csv',
    'phantom': 'results/phantom/phantom_per_image_metrics_2025-05-20_13-29-51.csv',
    'scd': 'results/SCD/SCD_per_image_metrics_2025-05-20_13-29-51.csv',
    'swfd': 'results/SWFD/SWFD_per_image_metrics_2025-05-20_13-29-51.csv',
    'v_phantom': 'results/v_phantom/v_phantom_per_image_metrics_2025-05-20_13-29-51.csv',
}

BOXPLOT_DIR = os.path.join(PROJECT_ROOT, "results/visualize/boxplots")
os.makedirs(BOXPLOT_DIR, exist_ok=True)

ttest_results = []
config_legend_map = defaultdict(dict)
all_data = []

# Load and combine per-image CSVs
for dataset_key, csv_path in PER_IMAGE_CSV_PATHS.items():
    if not os.path.exists(csv_path):
        print(f"[SKIP] {dataset_key} (missing file)")
        continue

    df = pd.read_csv(csv_path)
    df["Configuration"] = df["configuration"].astype(str).str.strip()

    # === Special case filtering ===
    if dataset_key == "msfd":
        df = df[df["Configuration"].str.contains("_w760")]
        df["Dataset"] = "msfd"
    elif dataset_key == "scd":
        df_ms = df[df["Configuration"].str.startswith("ms,")].copy()
        df_vc = df[df["Configuration"].str.startswith("vc,")].copy()
        df_ms["Dataset"] = "scd_ms"
        df_vc["Dataset"] = "scd_vc"
        all_data.extend([df_ms, df_vc])
        continue
    elif dataset_key == "swfd":
        df_ms = df[df["Configuration"].str.startswith("ms,")].copy()
        df_sc = df[df["Configuration"].str.startswith("sc,")].copy()
        df_ms["Dataset"] = "swfd_ms"
        df_sc["Dataset"] = "swfd_sc"
        all_data.extend([df_ms, df_sc])
        continue
    else:
        df["Dataset"] = dataset_key

    all_data.append(df)

# Merge all datasets
df_all = pd.concat(all_data, ignore_index=True)

# Metrics (exclude meta columns)
META_COLS = ["dataset", "configuration", "ground_truth", "wavelength", "image_path", "Dataset", "Configuration"]
all_metrics = [col for col in df_all.columns if col not in META_COLS]

# Assign standardized config labels per dataset
df_all["StandardConfig"] = None
for dataset in df_all["Dataset"].unique():
    if dataset in manual_config_sort:
        configs = manual_config_sort[dataset]
    else:
        configs = sorted(df_all[df_all["Dataset"] == dataset]["Configuration"].unique())
    
    for i, cfg in enumerate(configs):
        label = f"Subset {i+1}"
        df_all.loc[(df_all["Dataset"] == dataset) & (df_all["Configuration"] == cfg), "StandardConfig"] = label
        config_legend_map[dataset][label] = cfg

# === BOXPLOTS ===
for metric in all_metrics:
    plt.figure(figsize=(12, 6))
    plot_data = df_all.dropna(subset=[metric]).copy()
    plot_data["Dataset"] = plot_data["Dataset"].replace({"denoising": "NNE"})   
    sns.boxplot(x="Dataset", y=metric, hue="StandardConfig", data=plot_data, palette="Set2")
    plt.title(f"Distribution of {metric} per Dataset")
    plt.xticks(rotation=45)
    plt.ylabel(metric)

    handles, labels = plt.gca().get_legend_handles_labels()

    # Sort the legend entries numerically (e.g., Config 1, Config 2, ..., Config 10)
    def config_sort_key(label):
        match = re.search(r"(\d+)", label)
        return int(match.group(1)) if match else float('inf')

    sorted_pairs = sorted(zip(handles, labels), key=lambda x: config_sort_key(x[1]))
    sorted_handles, sorted_labels = zip(*sorted_pairs)

    plt.legend(sorted_handles, sorted_labels, title="Config", bbox_to_anchor=(1.01, 1), loc='upper left')
    
    plt.tight_layout()
    boxplot_path = os.path.join(BOXPLOT_DIR, f"{metric}_boxplot.png")
    plt.savefig(boxplot_path)
    plt.close()
    print(f"[INFO] Saved boxplot for {metric} → {boxplot_path}")

# Print legend mapping to console
print("\n[LEGEND: Standardized Config Mappings]")
for ds, mapping in config_legend_map.items():
    print(f"{ds}:")
    for label, cfg in mapping.items():
        print(f"  {label} = {cfg}")

# === T-TESTS ===
for metric in all_metrics:
    for dataset in df_all["Dataset"].unique():
        subset = df_all[df_all["Dataset"] == dataset].dropna(subset=[metric])

        # Get config mapping from your legend
        if dataset not in config_legend_map:
            print(f"[SKIP] No config_legend_map for {dataset}")
            continue

        config_map = config_legend_map[dataset]  # e.g., {Subset 1: config_name, ...}
        ordered_configs = list(config_map.values())

        # Only include configs that actually exist in the data for this metric
        existing_configs = subset["Configuration"].unique()
        valid_configs = [cfg for cfg in ordered_configs if cfg in existing_configs]

        # Pairwise comparisons in correct order
        for cfg1, cfg2 in combinations(valid_configs, 2):
            vals1 = subset[subset["Configuration"] == cfg1][metric]
            vals2 = subset[subset["Configuration"] == cfg2][metric]

            if len(vals1) < 2 or len(vals2) < 2:
                print(f"[SKIP] {dataset} - {metric} ({cfg1} vs {cfg2}): not enough samples")
                continue

            # Compute t-test
            stat, pval = ttest_ind(vals1, vals2, equal_var=False)
            mean1, mean2 = vals1.mean(), vals2.mean()
            var1, var2 = vals1.var(ddof=1), vals2.var(ddof=1)
            n1, n2 = len(vals1), len(vals2)

            # Store result
            ttest_results.append({
                "Metric": metric,
                "Dataset": dataset,
                "Config1": cfg1,
                "Config2": cfg2,
                "Mean1": mean1,
                "Mean2": mean2,
                "Var1": var1,
                "Var2": var2,
                "N1": n1,
                "N2": n2,
                "P-Value": pval
            })

# Save results
ttest_df = pd.DataFrame(ttest_results)
ttest_df.sort_values(by="P-Value", inplace=True)

output_path = os.path.join(PROJECT_ROOT, "results/visualize/metric_significance.csv")
ttest_df.to_csv(output_path, index=False)

print(f"[INFO] T-test results saved to: {output_path}")
print(ttest_df.head(10))

# --- Enhanced Ranking: Combine significance count with normalized effect size ---
# Filter for significant differences (p < 0.05)
significant_df = ttest_df[ttest_df["P-Value"] < 0.05].copy()

# Compute absolute difference between means
significant_df["AbsMeanDiff"] = (significant_df["Mean1"] - significant_df["Mean2"]).abs()

# Define expected value ranges for normalization
metric_ranges = {
    "BRISQUE": 100,
    "PSNR": 80,
    "SSIM": 1,
    "UQI": 1,
    "FSIM": 1,
    "VIF": 1,
    "GMSD": 0.35,
    "GMSD_norm": 1,
    "MSGMSD": 0.35,
    "MSGMSD_norm": 1,
    "IWSSIM": 1,
    "MSSSIM": 1,
    "S3IM": 1,
    "HAARPSI": 1,
    "CLIP-IQA": 1,
}

# Filter out any metrics not in the allowed list (e.g. TV)
significant_df = significant_df[significant_df["Metric"].isin(metric_ranges.keys())]

# Normalize abs mean difference per row
significant_df["NormAbsMeanDiff"] = significant_df.apply(
    lambda row: row["AbsMeanDiff"] / metric_ranges[row["Metric"]], axis=1
)

# Define metrics where lower is better
lower_is_better = ["BRISQUE", "GMSD", "GMSD_norm", "MSGMSD", "MSGMSD_norm"]

# Only keep rows where the direction of improvement is correct
significant_df = significant_df[
    significant_df.apply(
        lambda row: (row["Mean2"] > row["Mean1"]) if row["Metric"] not in lower_is_better else (row["Mean2"] < row["Mean1"]),
        axis=1
    )
]

# Compute average normalized diff and directional significance count per metric
mean_diffs = significant_df.groupby("Metric")["NormAbsMeanDiff"].mean()
sig_counts = significant_df.groupby("Metric").size()

# Combine into ranking DataFrame
ranking_df = pd.DataFrame({
    "SignificantCount": sig_counts,
    "NormMeanDiff": mean_diffs
})
ranking_df["CompositeScore"] = ranking_df["SignificantCount"] * ranking_df["NormMeanDiff"]

# Sort and save
ranking_df.sort_values("CompositeScore", ascending=False, inplace=True)
ranking_df.to_csv(os.path.join(PROJECT_ROOT, "results/visualize/top_metric_ranking.csv"))

print("\n[INFO] Top metrics ranked by normalized CompositeScore (significance × normalized mean diff):")
print(ranking_df.head(5))

# Step 1: Rename for clarity
df_all = df_all.rename(columns={"StandardConfig": "Subset"})
df_all["Subset"] = df_all["Subset"].astype(str)

def add_significance_between_first_last(ax, metric, dataset, plot_data, significance_df, offset=0.02):
    """
    Add a single significance annotation (e.g., *, **, ***) between the first and last subset,
    but only if *all adjacent subset comparisons* are significant (p < 0.05).
    The annotation reflects the weakest significance in the chain.
    """
    subset_data = plot_data[plot_data["Dataset"] == dataset]
    subset_order = sorted(subset_data["Subset"].unique())
    if len(subset_order) < 2:
        return

    subset1 = subset_order[0]
    subset2 = subset_order[-1]

    try:
        # Map internal names back to original dataset names in CSV
        dataset_lookup = {"NNE": "denoising", "EFA": "pa_experiment"}  # Extend if needed
        original_dataset = dataset_lookup.get(dataset, dataset)

        # Get actual config names from df_all
        cfg_rows = df_all[
            (df_all["Dataset"] == original_dataset) &
            (df_all["Subset"].isin(subset_order))
        ][["Subset", "Configuration"]].drop_duplicates()
        subset_to_config = dict(zip(cfg_rows["Subset"], cfg_rows["Configuration"]))

        # Helper to map p-values to stars
        def pval_to_star(p):
            if p < 0.001:
                return "***"
            elif p < 0.01:
                return "**"
            elif p < 0.05:
                return "*"
            elif p < 0.1:
                return "."
            return None

        # Check all adjacent pairs
        weakest_star = "***"
        for i in range(len(subset_order) - 1):
            s1, s2 = subset_order[i], subset_order[i + 1]
            c1, c2 = subset_to_config.get(s1), subset_to_config.get(s2)
            if not c1 or not c2:
                print(f"[WARN] Missing config for {s1} or {s2} in {dataset}")
                return

            row = significance_df[
                (significance_df["Metric"] == metric) &
                (significance_df["Dataset"] == original_dataset) &
                (
                    ((significance_df["Config1"] == c1) & (significance_df["Config2"] == c2)) |
                    ((significance_df["Config1"] == c2) & (significance_df["Config2"] == c1))
                )
            ]

            if row.empty:
                print(f"[DEBUG] No p-value for {c1} vs {c2} in {metric} ({dataset})")
                return

            pval = row["P-Value"].values[0]
            mean1 = row["Mean1"].values[0]
            mean2 = row["Mean2"].values[0]
            star = pval_to_star(pval)

            if star is None:
                return  # Not significant

            # Enforce directional check (e.g., higher is better)
            better_if_higher = metric not in ["BRISQUE", "GMSD", "GMSD_norm", "MSGMSD", "MSGMSD_norm"]  # lower is better for these
            is_direction_ok = (mean2 > mean1) if better_if_higher else (mean2 < mean1)

            if not is_direction_ok:
                print(f"[SKIP] Wrong direction: {metric} on {dataset} from {c1} to {c2}")
                return

            if len(star) < len(weakest_star):
                weakest_star = star  # Downgrade star level if weaker link found

        # Get max y value for line height
        values1 = subset_data[subset_data["Subset"] == subset1][metric]
        values2 = subset_data[subset_data["Subset"] == subset2][metric]
        y_max = max(values1.max(), values2.max())

        # Compute line positions
        dataset_pos = list(plot_data["Dataset"].unique()).index(dataset)
        num_subsets = len(subset_order)
        group_width = 0.8
        spacing = group_width / max(num_subsets - 1, 1)
        x1 = dataset_pos - group_width / 2
        x2 = dataset_pos - group_width / 2 + spacing * (num_subsets - 1)

        # Draw the annotation line and weakest star
        ax.plot([x1, x1, x2, x2],
                [y_max + offset, y_max + offset * 1.3, y_max + offset * 1.3, y_max + offset],
                lw=1.5, color='black')
        ax.text((x1 + x2) / 2, y_max + offset * 1.5, weakest_star,
                ha='center', va='bottom', fontsize=14)

    except Exception as e:
        print(f"[WARN] Could not annotate significance for {dataset} - {metric}: {e}")

# Step 3: Chunk into 2 figures of 7 rows each
filtered_metrics = [m for m in all_metrics if "_norm" not in m][:14]
chunks = [filtered_metrics[:7], filtered_metrics[7:14]]

for fig_idx, chunk in enumerate(chunks):
    fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(18, 30), sharex=False)
    axes = axes.flatten()

    for idx, metric in enumerate(chunk):
        ax = axes[idx]
        plot_data = df_all.dropna(subset=[metric]).copy()
        plot_data["Dataset"] = plot_data["Dataset"].replace({
            "denoising": "NNE",
            "pa_experiment": "EFA"
        })

        sns.boxplot(x="Dataset", y=metric, hue="Subset", data=plot_data,
                    palette="Set2", ax=ax, showfliers=True, width=0.7)

        # Panel label (e.g., A1 to G1)
        row_letter = chr(65 + idx + fig_idx * 7)
        ax.text(0.98, 0.93, f"{row_letter}", transform=ax.transAxes,
                ha="right", va="top", fontsize=15,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))

        ax.set_ylabel(metric, fontsize=20)
        ax.set_xlabel("")
        ax.set_title("")  # No title per box
        ax.tick_params(axis='y', labelsize=16)
        ax.get_legend().remove()

        significance_df = pd.read_csv("results/visualize/metric_significance.csv")
        for dataset in plot_data["Dataset"].unique():
            add_significance_between_first_last(ax, metric, dataset, plot_data, significance_df)

        if idx == 6:
            x_labels = plot_data["Dataset"].unique().tolist()
            ax.set_xticks(np.arange(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=45, fontsize=20, ha='right')
        else:
            ax.set_xticklabels([])

    # Shared Legend
    handles, labels = axes[0].get_legend_handles_labels()
    def subset_sort_key(label):
        match = re.search(r"(\d+)", label)
        return int(match.group(1)) if match else float('inf')
    sorted_pairs = sorted(zip(handles, labels), key=lambda x: subset_sort_key(x[1]))
    sorted_handles, sorted_labels = zip(*sorted_pairs)

    fig.legend(sorted_handles, sorted_labels, title="Subset", loc='lower center',
               ncol=min(8, len(sorted_labels)), fontsize=18, title_fontsize=22, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    fig_path = os.path.join(BOXPLOT_DIR, f"combined_metrics_boxplot_final_page{fig_idx+1}.png")
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved boxplot page {fig_idx+1}: {fig_path}")

for fig_idx, chunk in enumerate(chunks):
    fig_width = 50
    row_height = 5  # Empirical tuning; adjust if too squashed or too spaced
    fig_height = row_height * 7

    fig, axes = plt.subplots(nrows=7, ncols=1, figsize=(fig_width, fig_height), sharex=False)
    axes = axes.flatten()

    for idx, metric in enumerate(chunk):
        ax = axes[idx]
        plot_data = df_all.dropna(subset=[metric]).copy()
        plot_data["Dataset"] = plot_data["Dataset"].replace({
            "denoising": "NNE",
            "pa_experiment": "EFA"
        })

        sns.boxplot(x="Dataset", y=metric, hue="Subset", data=plot_data,
                    palette="Set2", ax=ax, showfliers=True, width=0.7)

        # Panel label (e.g., A1 to G1)
        row_letter = chr(65 + idx + fig_idx * 7)
        ax.text(0.98, 0.93, f"{row_letter}", transform=ax.transAxes,
                ha="right", va="top", fontsize=30,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="gray"))

        ax.set_ylabel(metric, fontsize=40)
        ax.set_xlabel("")
        ax.set_title("")
        ax.tick_params(axis='y', labelsize=40)
        ax.get_legend().remove()

        significance_df = pd.read_csv("results/visualize/metric_significance.csv")
        for dataset in plot_data["Dataset"].unique():
            add_significance_between_first_last(ax, metric, dataset, plot_data, significance_df)

        if idx == 6:
            x_labels = plot_data["Dataset"].unique().tolist()
            ax.set_xticks(np.arange(len(x_labels)))
            ax.set_xticklabels(x_labels, rotation=30, fontsize=50, ha='right')
        else:
            ax.set_xticklabels([])

    # Shared Legend
    handles, labels = axes[0].get_legend_handles_labels()
    def subset_sort_key(label):
        match = re.search(r"(\d+)", label)
        return int(match.group(1)) if match else float('inf')
    sorted_pairs = sorted(zip(handles, labels), key=lambda x: subset_sort_key(x[1]))
    sorted_handles, sorted_labels = zip(*sorted_pairs)

    fig.legend(sorted_handles, sorted_labels, title="Subset", loc='lower center',
               ncol=min(8, len(sorted_labels)), fontsize=36, title_fontsize=44, bbox_to_anchor=(0.5, -0.01))

    fig.tight_layout(rect=[0, 0.04, 1, 0.98])
    fig_path = os.path.join(BOXPLOT_DIR, f"combined_metrics_boxplot_FINALWIDE_page{fig_idx+1}.png")
    fig.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO]  Saved boxplot page {fig_idx+1} at fixed width {fig_width} inch → {fig_path}")