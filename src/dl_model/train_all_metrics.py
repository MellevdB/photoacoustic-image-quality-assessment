# src/dl_model/train_all_metrics.py
import os
from dl_model.train import train_model

# metrics_to_train = [
#     'SSIM', 'GMSD_norm', 'HAARPSI', 'IWSSIM','S3IM',
#     ['SSIM', 'GMSD_norm'],
#     ['SSIM', 'HAARPSI'],
#     ['GMSD_norm', 'HAARPSI'],
#     ['SSIM', 'GMSD_norm', 'HAARPSI', 'S3IM', 'IWSSIM']
# ]

# metrics_to_train = [
#     'SSIM', 'IWSSIM', 'HAARPSI', 'S3IM', 'GMSD_norm'
# ]

metrics_to_train = [
    'S3IM'
]


data_dir = "results"
device = "cuda"


# model_types = ["IQDCNN", "EfficientNetIQA", "best_model"]
model_types = ["EfficientNetIQA"]


def get_hparams_for_model(model_type: str):
    if model_type == "best_model":
        return {
            "dropout_rate": 0.0,
            "learning_rate": 1e-4,
            "num_fc_units": 128,
            "conv_filters": [32, 64, 128, 256],
            "loss_fn": "l1",
            "optimizer": "adam",
            "model_variant": "dropout",
        }
    elif model_type == "IQDCNN":
        return {
            "conv_filters": [32, 32, 32, 32],
            "num_fc_units": 1024,
            "dropout_rate": 0.3,
            "learning_rate": 1e-4,
            "loss_fn": "l1",
            "optimizer": "adam",
            "model_variant": "iqdcnn",
        }
    elif model_type == "EfficientNetIQA":
        return {
            "dropout_rate": 0.0,
            "learning_rate": 1e-4,
            "num_fc_units": 128,
            "conv_filters": [32, 64, 128, 256],
            "loss_fn": "l1",
            "optimizer": "adam",
            "model_variant": "efficientnet",
        }
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


split_tag = "shuffled_70_15_15"

for model_type in model_types:
    h = get_hparams_for_model(model_type)

    for metric in metrics_to_train:
        if isinstance(metric, str):
            print(f"\nTraining {model_type} model for metric: {metric}")
            model_dir = os.path.join("models", split_tag, model_type, metric)
            os.makedirs(model_dir, exist_ok=True)

            train_model(
                data_dir=data_dir,
                batch_size=128,
                learning_rate=h["learning_rate"],
                num_epochs=100,
                device=device,
                save_path=os.path.join(model_dir, "best_model.pth"),
                target_metric=metric,
                split_mode="shuffled",
                until_convergence=True,
                patience=10,
                dropout_rate=h["dropout_rate"],
                num_fc_units=h["num_fc_units"],
                conv_filters=h.get("conv_filters", [32, 64, 128, 256]),
                model_variant=h["model_variant"],
                loss_fn=h["loss_fn"],
                optimizer=h["optimizer"]
            )

        elif isinstance(metric, list):
            model_name = "_".join(metric)
            print(f"\nTraining {model_type} model for metrics: {metric}")
            model_dir = os.path.join("models", split_tag, model_type, model_name)
            os.makedirs(model_dir, exist_ok=True)

            multi_variant = (
                "multi" if model_type == "best_model"
                else "iqdcnn_multi" if model_type == "IQDCNN"
                else "efficientnet_multi" if model_type == "EfficientNetIQA"
                else None
            )
            if multi_variant is None:
                raise ValueError(f"Unsupported model_type for multi-metrics: {model_type}")

            train_model(
                data_dir=data_dir,
                batch_size=128,
                learning_rate=h["learning_rate"],
                num_epochs=100,
                device=device,
                save_path=os.path.join(model_dir, "best_model.pth"),
                target_metric=metric,
                split_mode="shuffled",
                until_convergence=True,
                patience=10,
                dropout_rate=h["dropout_rate"],
                num_fc_units=h["num_fc_units"],
                conv_filters=h.get("conv_filters", [32, 64, 128, 256]),
                model_variant=multi_variant,
                loss_fn=h["loss_fn"],
                optimizer=h["optimizer"]
            )
        else:
            raise ValueError(f"Invalid metric config: {metric}")