import os
import matplotlib.pyplot as plt
import pandas as pd
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, pearsonr
import torch
import torch.nn as nn
import numpy as np

from dl_model.inference import load_model_checkpoint, run_inference
from dl_model.utils import create_train_val_test_split

# metrics_to_eval = [
#     'SSIM', 'GMSD_norm', 'HAARPSI', 'IWSSIM','S3IM',
#     ['SSIM', 'GMSD_norm'],
#     ['SSIM', 'HAARPSI'],
#     ['GMSD_norm', 'HAARPSI'],
#     ['SSIM', 'GMSD_norm', 'HAARPSI', 'S3IM', 'IWSSIM']
# ]

metrics_to_eval = [
    ['SSIM', 'GMSD_norm', 'HAARPSI', 'S3IM', 'IWSSIM']
]



data_dir = "results"
device = "cuda"
configs = ["best_model", "IQDCNN", "EfficientNetIQA"]
_, _, test_sets = create_train_val_test_split(data_dir)
# test_sets = {"varied_split": test_sets["varied_split"]}

print("Going to evaluate the following metrics or metric combinations:")
print(metrics_to_eval)
print("For", configs)

for config in configs:
    for metric in metrics_to_eval:
        is_multi = isinstance(metric, list)
        metric_name = "_".join(metric) if is_multi else metric
        print(f"\nEvaluating [{config}] model trained on: {metric_name}")
        model_path = os.path.join("models", config, metric_name, "best_model.pth")
        model = load_model_checkpoint(model_path, device=device)
        criterion = nn.L1Loss()

        for dataset_name, test_data in test_sets.items():
            print(f" Testing on dataset: {dataset_name}")
            test_data.target_metric = metric
            test_loader = DataLoader(test_data, batch_size=16)

            model.eval()
            all_preds = []
            all_targets = []
            all_paths = []

            with torch.no_grad():
                for images, targets_batch, paths in test_loader:
                    images = images.to(device)
                    outputs = model(images).cpu().numpy()
                    targets_np = targets_batch.cpu().numpy()

                    all_preds.extend(outputs)
                    all_targets.extend(targets_np)
                    all_paths.extend(paths)

            preds = np.array(all_preds)
            targets = np.array(all_targets)
            if not is_multi:
                if preds.ndim == 2 and preds.shape[1] == 1:
                    preds = preds.squeeze(1)
                if targets.ndim == 2 and targets.shape[1] == 1:
                    targets = targets.squeeze(1)

            output_dir = os.path.join("results", "eval_model", config, metric_name, dataset_name)
            os.makedirs(output_dir, exist_ok=True)

            # === Scatter Plot ===
            if is_multi:
                num_outputs = len(metric)
                fig, axs = plt.subplots(1, num_outputs, figsize=(6*num_outputs, 5))
                if num_outputs == 1:
                    axs = [axs]
                for i in range(num_outputs):
                    axs[i].scatter(targets[:, i], preds[:, i], alpha=0.6, edgecolor='k')
                    axs[i].set_xlabel(f"True {metric[i]}")
                    axs[i].set_ylabel(f"Predicted {metric[i]}")
                    axs[i].set_title(f"{metric[i]} on {dataset_name}")
                    axs[i].set_xlim(0.0, 1.0)
                    axs[i].set_ylim(0.0, 1.0)
                    axs[i].set_xticks(np.arange(0.0, 1.1, 0.1))
                    axs[i].set_yticks(np.arange(0.0, 1.1, 0.1))
                    axs[i].grid(True, linestyle='--', linewidth=0.5)
                    axs[i].set_aspect('equal', adjustable='box')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"scatter_{metric_name}_{dataset_name}.png"))
                plt.close()
            else:
                plt.figure(figsize=(7, 6))
                plt.scatter(targets, preds, alpha=0.6, edgecolor='k')
                plt.xlabel(f"True {metric}")
                plt.ylabel(f"Predicted {metric}")
                plt.title(f"{metric} Prediction on {dataset_name} ({config})")
                plt.xlim(0.0, 1.0)
                plt.ylim(0.0, 1.0)
                plt.xticks(np.arange(0.0, 1.1, 0.1))
                plt.yticks(np.arange(0.0, 1.1, 0.1))
                plt.grid(True, linestyle='--', linewidth=0.5)
                plt.gca().set_aspect('equal', adjustable='box')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f"scatter_{metric}_{dataset_name}.png"))
                plt.close()

            # === Correlation ===
            corrs = []
            if is_multi:
                for i, m in enumerate(metric):
                    s_corr, _ = spearmanr(preds[:, i], targets[:, i])
                    p_corr, _ = pearsonr(preds[:, i], targets[:, i])
                    corrs.append((m, s_corr, p_corr))
                    print(f"{dataset_name} [{m}] → Spearman: {s_corr:.4f} | Pearson: {p_corr:.4f}")
            else:
                s_corr, _ = spearmanr(preds, targets)
                p_corr, _ = pearsonr(preds, targets)
                corrs.append((metric, s_corr, p_corr))
                print(f"{dataset_name} → Spearman: {s_corr:.4f} | Pearson: {p_corr:.4f}")

            # === Save CSV ===
            df = pd.DataFrame()
            df["image_path"] = [str(p).strip() for p in all_paths]
            if is_multi:
                for i, m in enumerate(metric):
                    df[f"target_{m}"] = targets[:, i]
                    df[f"prediction_{m}"] = preds[:, i]
            else:
                df["target"] = targets
                df["prediction"] = preds
            df.to_csv(os.path.join(output_dir, f"preds_vs_targets_{metric_name}_{dataset_name}.csv"), index=False)

            # === Save Correlation ===
            with open(os.path.join(output_dir, f"correlations_{dataset_name}.txt"), "w") as f:
                for m, s, p in corrs:
                    f.write(f"{m} → Spearman: {s:.4f} | Pearson: {p:.4f}\n")

            # === Loss Calculation ===
            l1_total_loss = 0.0
            mse_total_loss = 0.0
            mse_criterion = nn.MSELoss(reduction="none")

            with torch.no_grad():
                for images, labels, _ in test_loader:
                    images = images.to(device)
                    labels = labels.to(device).float()
                    if not is_multi:
                        labels = labels.unsqueeze(1)
                    outputs = model(images)
                    l1_loss = criterion(outputs, labels)
                    mse_loss = mse_criterion(outputs, labels).mean()
                    l1_total_loss += l1_loss.item() * images.size(0)
                    mse_total_loss += mse_loss.item() * images.size(0)

            avg_l1_loss = l1_total_loss / len(test_loader.dataset)
            avg_mse_loss = mse_total_loss / len(test_loader.dataset)
            print(f"{dataset_name} → Avg Test L1 Loss: {avg_l1_loss:.4f}")
            print(f"{dataset_name} → Avg Test MSE Loss: {avg_mse_loss:.4f}")

            with open(os.path.join(output_dir, f"test_loss_{dataset_name}.txt"), "w") as f:
                f.write(f"Test L1 Loss: {avg_l1_loss:.4f}\n")
                f.write(f"Test MSE Loss: {avg_mse_loss:.4f}\n")

        # === Loss Curve Plot ===
        loss_log_path = os.path.join("models", config, metric_name, "train_val_loss.csv")
        if os.path.exists(loss_log_path):
            loss_df = pd.read_csv(loss_log_path)
            plt.figure(figsize=(8, 5))
            plt.plot(loss_df["epoch"], loss_df["train_loss"], label="Train Loss", marker="o")
            plt.plot(loss_df["epoch"], loss_df["val_loss"], label="Val Loss", marker="x")
            plt.title(f"Train vs Val Loss for {metric_name} ({config})")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            loss_plot_path = os.path.join("results", "eval_model", config, metric_name, "train_val_loss_plot.png")
            os.makedirs(os.path.dirname(loss_plot_path), exist_ok=True)
            plt.savefig(loss_plot_path)
            plt.close()
            print(f" Saved loss plot to {loss_plot_path}")
        else:
            print(f" No train_val_loss.csv found for {metric_name} ({config}), skipping loss plot.")