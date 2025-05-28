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
#     'CLIP-IQA', 'SSIM', 'PSNR_norm', 'VIF', 'GMSD_norm', 'HAARPSI', 'MSSSIM', 'IWSSIM',
#     'MSGMSD_norm', 'BRISQUE_norm', 'TV'
# ]

metrics_to_eval = [
    "FSIM", "UQI", "S3IM"
]

# metrics_to_eval = [
#     'PSNR_norm', 'VIF', 'GMSD_norm', 'HAARPSI', 'MSSSIM'
# ]

data_dir = "results"
device = "cuda"

# Load test sets per dataset
_, _, test_sets = create_train_val_test_split(data_dir)

configs = ["best_config", "iqadcnn_config"]

for config in configs:
    for metric in metrics_to_eval:
        print(f"\n Evaluating [{config}] model trained on: {metric}")
        model_path = os.path.join("models", config, metric, "best_model.pth")
        model = load_model_checkpoint(model_path, device=device)
        criterion = nn.L1Loss()

        for dataset_name, test_data in test_sets.items():
            print(f" Testing on dataset: {dataset_name}")
            test_data.target_metric = metric
            test_loader = DataLoader(test_data, batch_size=16)

            preds = run_inference(model, test_loader, device=device)
            print(f" Prediction stats → mean: {np.mean(preds):.4f}, std: {np.std(preds):.4f}")
            targets = [label.item() for _, label in test_data]

            # Save dir: results/eval_model/<config>/<metric>/<dataset>
            output_dir = os.path.join("results", "eval_model", config, metric, dataset_name)
            os.makedirs(output_dir, exist_ok=True)

            # Scatter plot
            plt.figure(figsize=(7, 6))
            plt.scatter(targets, preds, alpha=0.6, edgecolor='k')
            plt.xlabel(f"True {metric}")
            plt.ylabel(f"Predicted {metric}")
            plt.title(f"{metric} Prediction on {dataset_name} ({config})")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"scatter_{metric}_{dataset_name}.png"))
            plt.close()

            # Correlation
            spearman_corr, _ = spearmanr(preds, targets)
            pearson_corr, _ = pearsonr(preds, targets)
            print(f"{dataset_name} → Spearman: {spearman_corr:.4f} | Pearson: {pearson_corr:.4f}")

            # Save CSV of predictions
            df = pd.DataFrame({
                "target": targets,
                "prediction": preds
            })
            df.to_csv(os.path.join(output_dir, f"preds_vs_targets_{metric}_{dataset_name}.csv"), index=False)

            # Save correlation summary
            with open(os.path.join(output_dir, f"correlations_{dataset_name}.txt"), "w") as f:
                f.write(f"Spearman: {spearman_corr:.4f}\n")
                f.write(f"Pearson : {pearson_corr:.4f}\n")

            # Final test loss
            model.eval()
            test_loss = 0.0
            with torch.no_grad():
                for images, labels in test_loader:
                    images = images.to(device)
                    labels = labels.to(device).unsqueeze(1).float()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    test_loss += loss.item() * images.size(0)
            test_loss /= len(test_loader.dataset)
            print(f"{dataset_name} → Test Loss (L1): {test_loss:.4f}")
            with open(os.path.join(output_dir, f"test_loss_{dataset_name}.txt"), "w") as f:
                f.write(f"Test L1 Loss: {test_loss:.4f}\n")

        # Plot loss curve if available
        loss_log_path = os.path.join("models", config, metric, "train_val_loss.csv")
        if os.path.exists(loss_log_path):
            loss_df = pd.read_csv(loss_log_path)
            plt.figure(figsize=(8, 5))
            plt.plot(loss_df["epoch"], loss_df["train_loss"], label="Train Loss", marker="o")
            plt.plot(loss_df["epoch"], loss_df["val_loss"], label="Val Loss", marker="x")
            plt.title(f"Train vs Val Loss for {metric} ({config})")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            loss_plot_path = os.path.join("results", "eval_model", config, metric, "train_val_loss_plot.png")
            os.makedirs(os.path.dirname(loss_plot_path), exist_ok=True)
            plt.savefig(loss_plot_path)
            plt.close()
            print(f" Saved loss plot to {loss_plot_path}")
        else:
            print(f" No train_val_loss.csv found for {metric} ({config}), skipping loss plot.")