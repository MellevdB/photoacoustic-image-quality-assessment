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
    'KneeSlice1': ['PA2', 'PA3', 'PA4', 'PA5', 'PA6', 'PA7'],
    'Phantoms': ['PA2', 'PA3', 'PA4', 'PA5', 'PA6', 'PA7'],
    'Transducers': ['PA2', 'PA3', 'PA4', 'PA5', 'PA6', 'PA7'],
    'SmallAnimal': ['PA2', 'PA3', 'PA4', 'PA5', 'PA6', 'PA7']
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

# Absolute project root (2 levels up from this script)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))


# Change accordingly
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
NR_TOP = ["BRISQUE", "TV"]
NR_BOTTOM = ["CLIP-IQA"]
FR_TOP = ["PSNR"]
FR_BOTTOM = [m for m in FR_METRICS if m != "PSNR"]

colors = {
    "FSIM": "b", "UQI": "g", "PSNR": "r", "SSIM": "c", "VIF": "m", "S3IM": "y",
    "MSSSIM": "tab:blue", "IWSSIM": "tab:orange", "GMSD": "tab:green",
    "MSGMSD": "tab:red", "HAARPSI": "tab:purple",
    "BRISQUE": "gray", "TV": "darkgreen", "CLIP-IQA": "gold"
}


def load_dataset_results(result_file):
    with open(result_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    # Remove dashed line separator
    lines = [line for line in lines if not re.fullmatch(r"-+", line)]

    # Parse header separately
    raw_header = shlex.split(lines[0])
    
    # Manually merge 'Ground' and 'Truth' back into one column
    fixed_header = []
    i = 0
    while i < len(raw_header):
        if raw_header[i] == "Ground" and i + 1 < len(raw_header) and raw_header[i + 1] == "Truth":
            fixed_header.append("Ground Truth")
            i += 2  # skip both "Ground" and "Truth"
        else:
            fixed_header.append(raw_header[i])
            i += 1

    # Now parse the data rows
    data_rows = []
    for i, line in enumerate(lines[1:]):
        if not line.strip(): continue
        row = shlex.split(line)
        if len(row) != len(fixed_header):
            print(f"[WARNING] Row {i} has {len(row)} columns but expected {len(fixed_header)}: {row}")
        data_rows.append(row)

    df = pd.DataFrame(data_rows, columns=fixed_header)

    # Convert all columns where possible
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass

    return df

def plot_dual_metrics(configs, metric_groups, title, filename):
    x = np.arange(len(configs))
    fig, (ax_top, ax_bottom) = plt.subplots(2, 1, sharex=True, figsize=(12, 8),
                                            gridspec_kw={'height_ratios': [1, 1]})

    for group, metrics, axis in [
        ("top", metric_groups['top'], ax_top),
        ("bottom", metric_groups['bottom'], ax_bottom)
    ]:
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
        # Set custom y-axis ranges
    top_metrics = list(metric_groups['top'].keys())
    bottom_metrics = list(metric_groups['bottom'].keys())

    if any(m in ["PSNR"] for m in top_metrics):
        ax_top.set_ylim(0, 80)
    elif any(m in ["BRISQUE", "TV"] for m in top_metrics):
        ax_top.set_ylim(0, 100)
    else:
        ax_top.set_ylim(0, 1)

    if any(m in ["BRISQUE", "TV"] for m in bottom_metrics):
        ax_bottom.set_ylim(0, 100)
    else:
        ax_bottom.set_ylim(0, 1)
    plt.tight_layout()
    output_dir = os.path.join(PROJECT_ROOT, "results/visualize/plots")
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, filename)
    plt.savefig(filename)
    plt.close()

def extract_metrics(df, metrics, config_list):
    if df.empty:
        raise ValueError("DataFrame is empty — possibly invalid result file format or read error.")

    config_col = "Configuration"
    df = df.copy()
    df[config_col] = df[config_col].astype(str).str.strip() 
    config_list = [cfg.strip() for cfg in config_list]

    # Filter and keep exact order from config_list
    if not set(config_list).intersection(set(df[config_col])):
        print("\n[WARNING] No configs matched.")
        print("[Available configs]:", df[config_col].unique())
        print("[Expected configs]:", config_list)
        raise ValueError("No matching configs found in result file.")

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

# === MAIN LOOP ===

# # DEBUG: show raw file and first rows
# result_path = DATASET_results_paths['denoising']
# print(f"\n[INFO] Reading: {result_path}")
# print("[INFO] Raw content preview:")
# with open(result_path, "r") as f:
#     for i, line in enumerate(f.readlines()):
#         print(line.strip())
#         if i > 10: break

# # Load DataFrame
# df = load_dataset_results(result_path)
# print("[INFO] Parsed DataFrame preview:")
# print(df.head())
# print("[INFO] Available configs:", df["Configuration"].tolist())

for dataset_key, result_path in DATASET_results_paths.items():
    print("Dataset key: ", dataset_key)
    if not os.path.exists(result_path):
        print(f"Skipping {dataset_key} (missing file)")
        continue

    df = load_dataset_results(result_path)
    configs_entry = DATASET_CONFIGS[dataset_key]

    # === SPECIAL CASE for pa_experiment ===
    if dataset_key == "pa_experiment":
        print("Handling pa_experiment subgroups...")
        df["Subgroup"] = df["Dataset"].apply(lambda x: x.split("/")[-1].strip())

        for subgroup, config_list in configs_entry.items():
            df_sub = df[df["Subgroup"] == subgroup].copy()
            if df_sub.empty:
                print(f"[WARNING] No rows matched subgroup {subgroup}")
                continue

            df_sub["Configuration"] = df_sub["Configuration"].astype(str).str.strip()

            fr_top_metrics = extract_metrics(df_sub, FR_TOP, config_list)
            fr_bot_metrics = extract_metrics(df_sub, FR_BOTTOM, config_list)
            nr_top_metrics = extract_metrics(df_sub, NR_TOP, config_list)
            nr_bot_metrics = extract_metrics(df_sub, NR_BOTTOM, config_list)

            plot_dual_metrics(
                config_list,
                {"top": fr_top_metrics, "bottom": fr_bot_metrics},
                f"{dataset_key.upper()} - {subgroup} (FR)",
                f"{dataset_key}_{subgroup}_FR.png"
            )
            plot_dual_metrics(
                config_list,
                {"top": nr_top_metrics, "bottom": nr_bot_metrics},
                f"{dataset_key.upper()} - {subgroup} (NR)",
                f"{dataset_key}_{subgroup}_NR.png"
            )

    # === ALL OTHER DATASETS ===
    elif isinstance(configs_entry, dict):
        print("Handling multi-config (e.g., MS/VC) dataset...")
        for subcat, config_list in configs_entry.items():
            df_sub = df[df["Configuration"].isin(config_list)]

            if df_sub.empty:
                print(f"[WARNING] No configs found for {dataset_key} - {subcat}")
                continue

            fr_top_metrics = extract_metrics(df_sub, FR_TOP, config_list)
            fr_bot_metrics = extract_metrics(df_sub, FR_BOTTOM, config_list)
            nr_top_metrics = extract_metrics(df_sub, NR_TOP, config_list)
            nr_bot_metrics = extract_metrics(df_sub, NR_BOTTOM, config_list)

            plot_dual_metrics(
                config_list,
                {"top": fr_top_metrics, "bottom": fr_bot_metrics},
                f"{dataset_key.upper()} - {subcat} (FR)",
                f"{dataset_key}_{subcat}_FR.png"
            )
            plot_dual_metrics(
                config_list,
                {"top": nr_top_metrics, "bottom": nr_bot_metrics},
                f"{dataset_key.upper()} - {subcat} (NR)",
                f"{dataset_key}_{subcat}_NR.png"
            )

    else:
        print("Handling single-config dataset...")
        config_list = configs_entry

        fr_top_metrics = extract_metrics(df, FR_TOP, config_list)
        fr_bot_metrics = extract_metrics(df, FR_BOTTOM, config_list)
        nr_top_metrics = extract_metrics(df, NR_TOP, config_list)
        nr_bot_metrics = extract_metrics(df, NR_BOTTOM, config_list)

        plot_dual_metrics(
            config_list,
            {"top": fr_top_metrics, "bottom": fr_bot_metrics},
            f"{dataset_key.upper()} (FR)",
            f"{dataset_key}_FR.png"
        )
        plot_dual_metrics(
            config_list,
            {"top": nr_top_metrics, "bottom": nr_bot_metrics},
            f"{dataset_key.upper()} (NR)",
            f"{dataset_key}_NR.png"
        )

for dataset_key, result_path in DATASET_results_paths.items():
    if dataset_key != "denoising":
        continue  # Only run for denoising

    print("Dataset key: ", dataset_key)
    if not os.path.exists(result_path):
        print(f"Skipping {dataset_key} (missing file)")
        continue

    df = load_dataset_results(result_path)
    config_list = DATASET_CONFIGS[dataset_key]

    fr_top_metrics = extract_metrics(df, FR_TOP, config_list)
    fr_bot_metrics = extract_metrics(df, FR_BOTTOM, config_list)
    nr_top_metrics = extract_metrics(df, NR_TOP, config_list)
    nr_bot_metrics = extract_metrics(df, NR_BOTTOM, config_list)

    # === DEBUG PRINT ===
    print("\n[DEBUG] Metric scores per configuration:")
    for i, config in enumerate(config_list):
        print(f"  Config: {config}")
        for metric_group, label in zip([fr_top_metrics, fr_bot_metrics, nr_top_metrics, nr_bot_metrics],
                                       ['FR_TOP', 'FR_BOTTOM', 'NR_TOP', 'NR_BOTTOM']):
            for metric, values in metric_group.items():
                mean_val = values['mean'][i]
                std_val = values['std'][i]
                print(f"    [{label}] {metric}: mean={mean_val:.4f}, std={std_val:.4f}")
