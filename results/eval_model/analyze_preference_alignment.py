import pandas as pd
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

# === CONFIG ===
models = ["best_model", "IQDCNN", "EfficientNetIQA"]
metrics_to_eval = [
    'SSIM', 'GMSD_norm', 'HAARPSI', 'IWSSIM', 'S3IM',
    ['SSIM', 'GMSD_norm'],
    ['SSIM', 'HAARPSI'],
    ['GMSD_norm', 'HAARPSI'],
    ['SSIM', 'GMSD_norm', 'HAARPSI', 'S3IM', 'IWSSIM']
]

survey_csv = "results/varied_split/results17062025.csv"
base_result_dir = "results/eval_model"

# === FUNCTIONS ===

def load_expert_comparisons(csv_path):
    df = pd.read_csv(csv_path)
    all_comparisons = []
    for comparisons_str in df["comparisons"]:
        comparisons = json.loads(comparisons_str.replace('""', '"'))  # Fix double quotes
        for comp in comparisons:
            all_comparisons.append({
                "scene": comp["sceneId"].strip(),
                "left": comp["left"].strip(),
                "right": comp["right"].strip(),
                "winner": comp["winner"].strip()
            })
    return pd.DataFrame(all_comparisons)

def load_predictions(pred_csv_path):
    df = pd.read_csv(pred_csv_path)

    # Extract scene ID from path
    df["filename"] = df["image_path"].apply(lambda p: os.path.basename(str(p)).strip())
    df["scene"] = df["image_path"].apply(lambda p: str(p).split("scene_")[-1].split("/")[0] if "scene_" in str(p) else "unknown")
    df["key"] = "scene_" + df["scene"] + "/" + df["filename"]
    df.columns = [col.strip() for col in df.columns]

    return df.set_index("key")

# def evaluate_pairwise_accuracy(comp_df, pred_df, metric_col):
#     correct = 0
#     total = 0
#     missing = 0

#     print(f"\n[DEBUG] First comparison: left={comp_df.iloc[0]['left']}, right={comp_df.iloc[0]['right']}")
#     print(f"[DEBUG] Available predictions (sample): {list(pred_df.index[:5])} ... total={len(pred_df)}")

#     for _, row in comp_df.iterrows():
#         left_key = f"{row['scene']}/{row['left']}"
#         right_key = f"{row['scene']}/{row['right']}"
#         winner_key = f"{row['scene']}/{row['winner']}"

#         if left_key not in pred_df.index or right_key not in pred_df.index:
#             print(f"[MISSING] One or both not in predictions → left: {left_key in pred_df.index}, right: {right_key in pred_df.index} (left={left_key}, right={right_key})")
#             missing += 1
#             continue

#         score_left = pred_df.loc[left_key, metric_col]
#         score_right = pred_df.loc[right_key, metric_col]
#         predicted_winner = left_key if score_left > score_right else right_key
#         if predicted_winner == winner_key:
#             correct += 1
#         total += 1

#     acc = correct / total if total > 0 else 0.0
#     return acc, correct, total, missing

# def evaluate_for_model_and_metric(model, metric):
#     metric_name = "_".join(metric) if isinstance(metric, list) else metric
#     metric_cols = [f"prediction_{m}" for m in metric] if isinstance(metric, list) else ["prediction"]
#     csv_path = os.path.join(base_result_dir, model, metric_name, "varied_split", f"preds_vs_targets_{metric_name}_varied_split.csv")

#     if not os.path.exists(csv_path):
#         print(f"[SKIP] File missing: {csv_path}")
#         return None

#     pred_df = load_predictions(csv_path)
#     comp_df = load_expert_comparisons(survey_csv)

#     # Combine metric columns if needed
#     if len(metric_cols) > 1:
#         pred_df["prediction_combo"] = pred_df[metric_cols].mean(axis=1)
#         metric_col = "prediction_combo"
#     else:
#         metric_col = metric_cols[0]

#     # Evaluate overall
#     overall_acc, correct, total, missing = evaluate_pairwise_accuracy(comp_df, pred_df, metric_col)

#     # Evaluate scene-wise
#     scene_stats = []
#     for scene_id, group in comp_df.groupby("scene"):
#         acc, corr, tot, _ = evaluate_pairwise_accuracy(group, pred_df, metric_col)
#         scene_stats.append({
#             "model": model,
#             "metric": metric_name,
#             "scene": scene_id,
#             "accuracy": acc,
#             "correct": corr,
#             "total": tot
#         })

#     summary = {
#         "model": model,
#         "metric": metric_name,
#         "overall_accuracy": overall_acc,
#         "correct": correct,
#         "total": total,
#         "missing": missing,
#     }

#     return summary, scene_stats

# # === MAIN RUN ===

# all_summaries = []
# all_scene_stats = []

# for model in models:
#     for metric in metrics_to_eval:
#         result = evaluate_for_model_and_metric(model, metric)
#         if result:
#             summary, scenes = result
#             all_summaries.append(summary)
#             all_scene_stats.extend(scenes)
#             print(f"[DONE] {summary['model']} | {summary['metric']} → Acc: {summary['overall_accuracy']:.4f} ({summary['correct']}/{summary['total']})")

# # === SAVE RESULTS ===

# summary_df = pd.DataFrame(all_summaries)
# scene_df = pd.DataFrame(all_scene_stats)

# summary_path = os.path.join(base_result_dir, "preference_alignment_summary.csv")
# scene_path = os.path.join(base_result_dir, "preference_alignment_scene_breakdown.csv")

# summary_df.to_csv(summary_path, index=False)
# scene_df.to_csv(scene_path, index=False)

# print(f"\n[✓] Saved summary to {summary_path}")
# print(f"[✓] Saved scene breakdown to {scene_path}")

# # === VISUALIZATION ===

# plot_dir = os.path.join(base_result_dir, "plots")
# os.makedirs(plot_dir, exist_ok=True)

# # Plot 1: Overall preference alignment barplot
# plt.figure(figsize=(12, 6))
# plot_df = summary_df.copy()
# plot_df["metric"] = plot_df["metric"].str.replace("_", "\n", regex=False)

# sns.barplot(data=plot_df, x="metric", y="overall_accuracy", hue="model")
# plt.ylabel("Preference Alignment Accuracy")
# plt.xlabel("Metric (or Metric Combination)")
# plt.ylim(0.0, 1.0)
# plt.title("Overall Preference Accuracy by Model and Metric")
# plt.legend(title="Model", loc="lower right")
# plt.grid(True, axis='y', linestyle='--', linewidth=0.5)
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.savefig(os.path.join(plot_dir, "overall_preference_accuracy.png"))
# plt.savefig(os.path.join(plot_dir, "overall_preference_accuracy.pdf"))
# plt.close()

# # Plot 2: Scene-wise heatmaps per metric
# scene_df_copy = scene_df.copy()
# scene_df_copy["metric"] = scene_df_copy["metric"].str.replace("_", "\n", regex=False)

# for metric in scene_df_copy["metric"].unique():
#     plt.figure(figsize=(8, 4))
#     data = scene_df_copy[scene_df_copy["metric"] == metric]
#     heatmap_data = data.pivot(index="model", columns="scene", values="accuracy")

#     sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="Blues", vmin=0.0, vmax=1.0, cbar_kws={"label": "Accuracy"})
#     plt.title(f"Scene-wise Preference Accuracy\nMetric: {metric}")
#     plt.ylabel("Model")
#     plt.xlabel("Scene")
#     plt.tight_layout()

#     save_name = f"scenewise_preference_accuracy_{metric.replace(chr(10), '_')}"
#     plt.savefig(os.path.join(plot_dir, f"{save_name}.png"))
#     plt.savefig(os.path.join(plot_dir, f"{save_name}.pdf"))
#     plt.close()

# === VISUALIZATION (Improved Font Sizes for Overleaf) ===
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# Load from saved CSV
summary_df = pd.read_csv(os.path.join(base_result_dir, "preference_alignment_summary.csv"))
plot_dir = os.path.join(base_result_dir, "plots")
os.makedirs(plot_dir, exist_ok=True)

# Rename models for legend
summary_df["model"] = summary_df["model"].replace({"best_model": "PhotoacousticQualityNet"})
summary_df["metric"] = summary_df["metric"].str.replace("GMSD_norm", "GMSD")

# Format metric names
def format_metric_name(m):
    if m == "SSIM_GMSD_norm_HAARPSI_S3IM_IWSSIM":
        return "All five metrics"
    parts = m.split("_")
    parts = ["GMSD" if p == "GMSD_norm" else p for p in parts]
    return "\n".join(parts) if len(parts) > 1 else parts[0]

summary_df["metric_display"] = summary_df["metric"].apply(format_metric_name)

# Plot
sns.set(style="whitegrid", font_scale=1.6)

plt.figure(figsize=(14, 8))
sns.barplot(data=summary_df, x="metric_display", y="overall_accuracy", hue="model")

plt.ylabel("Preference Alignment Accuracy", fontsize=20)
plt.xlabel("Metric (or Metric Combination)", fontsize=20)
plt.ylim(0.0, 1.0)
plt.title("Overall Preference Accuracy by Model and Metric", fontsize=22)
plt.legend(title="Model", loc="lower right", fontsize=16, title_fontsize=18)
plt.grid(True, axis='y', linestyle='--', linewidth=0.5)

plt.xticks(rotation=0, ha="center", fontsize=16)
plt.yticks(fontsize=16)

plt.tight_layout()
plt.savefig(os.path.join(plot_dir, "overall_preference_accuracy_cleaned.png"))
plt.savefig(os.path.join(plot_dir, "overall_preference_accuracy_cleaned.pdf"))
plt.close()

# # --- Plot 2: Scene-wise heatmaps ---
# scene_df_copy = scene_df.copy()
# scene_df_copy["metric"] = scene_df_copy["metric"].str.replace("_", "\n", regex=False)

# for metric in scene_df_copy["metric"].unique():
#     plt.figure(figsize=(10, 6))
#     data = scene_df_copy[scene_df_copy["metric"] == metric]
#     heatmap_data = data.pivot(index="model", columns="scene", values="accuracy")

#     sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="Blues", vmin=0.0, vmax=1.0,
#                 cbar_kws={"label": "Accuracy"}, annot_kws={"fontsize": 14})
#     plt.title(f"Scene-wise Preference Accuracy\nMetric: {metric}", fontsize=20)
#     plt.ylabel("Model", fontsize=18)
#     plt.xlabel("Scene", fontsize=18)
#     plt.xticks(rotation=45, ha="right", fontsize=14)
#     plt.yticks(fontsize=14)
#     plt.tight_layout()

#     save_name = f"scenewise_preference_accuracy_{metric.replace(chr(10), '_')}_larger_fonts"
#     plt.savefig(os.path.join(plot_dir, f"{save_name}.png"))
#     plt.savefig(os.path.join(plot_dir, f"{save_name}.pdf"))
#     plt.close()