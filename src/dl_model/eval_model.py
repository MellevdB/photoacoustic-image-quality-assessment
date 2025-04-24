# src/dl_model/eval_all_models.py
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from scipy.stats import spearmanr, pearsonr
from dl_model.inference import load_model_checkpoint, run_inference
from dl_model.utils import create_train_val_splits, PhotoacousticDatasetFromDataFrame

metrics_to_eval = [
    'CLIP-IQA', 'SSIM', 'PSNR', 'VIF', 'GMSD', 'HAARPSI', 'MSSSIM', 'IWSSIM',
    'MSGMSD', 'BRISQUE', 'TV'
]

data_dir = "results"
device = "cuda"

# Load common test split once
_, _, test_data = create_train_val_splits(data_dir)
for metric in metrics_to_eval:
    print(f"\n🔍 Evaluating model trained on: {metric}")
    # Adapt test dataset to target this metric
    test_data.target_metric = metric
    test_loader = DataLoader(test_data, batch_size=16)

    model_path = os.path.join("models", metric, "best_model.pth")
    model = load_model_checkpoint(model_path, device=device)

    preds = run_inference(model, test_loader, device=device)
    targets = [label.item() for _, label in test_data]

    # Output folder
    output_dir = os.path.join("results", "eval_model", metric)
    os.makedirs(output_dir, exist_ok=True)

    # Scatter plot
    plt.figure(figsize=(7, 6))
    plt.scatter(targets, preds, alpha=0.6, edgecolor='k')
    plt.xlabel(f"True {metric}")
    plt.ylabel(f"Predicted {metric}")
    plt.title(f"{metric} Prediction vs Ground Truth")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"scatter_{metric}.png"))
    plt.close()

    # Correlations
    spearman_corr, _ = spearmanr(preds, targets)
    pearson_corr, _ = pearsonr(preds, targets)
    print(f"Spearman: {spearman_corr:.4f} | Pearson: {pearson_corr:.4f}")

    # Save CSV
    import pandas as pd
    df = pd.DataFrame({
        "target": targets,
        "prediction": preds
    })
    df.to_csv(os.path.join(output_dir, f"preds_vs_targets_{metric}.csv"), index=False)

    # Save correlations
    with open(os.path.join(output_dir, f"correlations.txt"), "w") as f:
        f.write(f"Spearman: {spearman_corr:.4f}\n")
        f.write(f"Pearson : {pearson_corr:.4f}\n")