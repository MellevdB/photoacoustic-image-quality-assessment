# src/dl_model/train_all_metrics.py
import os
from dl_model.train import train_model

# metrics_to_train = [
#     'CLIP-IQA', 'SSIM', 'PSNR_norm', 'VIF', 'GMSD_norm', 'HAARPSI', 'MSSSIM', 'IWSSIM',
#     'MSGMSD_norm', 'BRISQUE_norm', 'TV'
# ]

metrics_to_train = [
    'UQI', 'S3IM'
]

data_dir = "results"
device = "cuda"

for metric in metrics_to_train:
    print(f"\n[CONFIG 1] Training best CV model for metric: {metric}")
    model_dir = os.path.join("models", "best_config", metric)
    os.makedirs(model_dir, exist_ok=True)
    train_model(
        data_dir=data_dir,
        batch_size=16,
        learning_rate=1e-4,
        num_epochs=100,
        device=device,
        save_path=os.path.join(model_dir, "best_model.pth"),
        target_metric=metric,
        until_convergence=True,
        patience=10,
        dropout_rate=0.0,
        num_fc_units=128
    )

    print(f"\n[CONFIG 2] Training IQA-DCNN config for metric: {metric}")
    model_dir = os.path.join("models", "iqadcnn_config", metric)
    os.makedirs(model_dir, exist_ok=True)
    train_model(
        data_dir=data_dir,
        batch_size=16,
        learning_rate=5e-5,
        num_epochs=100,
        device=device,
        save_path=os.path.join(model_dir, "best_model.pth"),
        target_metric=metric,
        until_convergence=True,
        patience=10,
        dropout_rate=0.3,
        num_fc_units=128
    )