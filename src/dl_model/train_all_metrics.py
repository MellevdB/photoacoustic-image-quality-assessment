# src/dl_model/train_all_metrics.py
import os
from dl_model.train import train_model

metrics_to_train = [
    'CLIP-IQA', 'SSIM', 'PSNR', 'VIF', 'GMSD', 'HAARPSI', 'MSSSIM', 'IWSSIM',
    'MSGMSD', 'BRISQUE', 'TV'
]

data_dir = "results"
device = "cuda"

for metric in metrics_to_train:
    print(f"\n🔁 Training model on metric: {metric}")
    model_dir = os.path.join("models", metric)
    os.makedirs(model_dir, exist_ok=True)

    train_model(
        data_dir=data_dir,
        batch_size=16,
        learning_rate=1e-3,
        num_epochs=15,
        device=device,
        save_path=os.path.join(model_dir, "best_model.pth"),
        target_metric=metric
    )