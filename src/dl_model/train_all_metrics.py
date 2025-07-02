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

metrics_to_train = [
    ['SSIM', 'HAARPSI']
]


data_dir = "results"
device = "cuda"


# Choose model type: "best_model", "IQDCNN", or "EfficientNetIQA"
# model_type = "best_model"
# model_type = "IQDCNN" 
model_type = "EfficientNetIQA" 

# Model-specific hyperparameters
if model_type == "best_model":
    dropout_rate = 0.0
    learning_rate = 1e-4
    num_fc_units = 128
    conv_filters=[32, 64, 128, 256]
    loss_fn = "huber"
    optimizer="adam"
elif model_type == "IQDCNN":
    conv_filters = [32, 32, 32, 32]
    num_fc_units = 1024
    dropout_rate = 0.3
    learning_rate = 5e-5
    loss_fn = "l1"
    optimizer = "adam"
elif model_type == "EfficientNetIQA":
    dropout_rate = 0.0 
    learning_rate = 2e-5
    num_fc_units = 128
    conv_filters=[32, 64, 128, 256]
    loss_fn = "mse"
    optimizer="adam"
else:
    raise ValueError(f"Unknown model_type: {model_type}")


for metric in metrics_to_train:
    if isinstance(metric, str):
        print(f"\nTraining {model_type} model for metric: {metric}")
        model_dir = os.path.join("models", model_type, metric)
        os.makedirs(model_dir, exist_ok=True)

        train_model(
            data_dir=data_dir,
            batch_size=16,
            learning_rate=learning_rate,
            num_epochs=100,
            device=device,
            save_path=os.path.join(model_dir, "best_model.pth"),
            target_metric=metric,
            until_convergence=True,
            patience=10,
            dropout_rate=dropout_rate,
            num_fc_units=num_fc_units,
            conv_filters=conv_filters,
            model_variant=(
                "dropout" if model_type == "best_model"
                else "iqdcnn" if model_type == "IQDCNN"
                else "efficientnet" if model_type == "EfficientNetIQA"
                else ValueError(f"Unsupported model_type for metrics: {model_type}")
            ),
            loss_fn=loss_fn,
            optimizer=optimizer
        )

    elif isinstance(metric, list):
        model_name = "_".join(metric)
        print(f"\nTraining {model_type} model for metrics: {metric}")
        model_dir = os.path.join("models", model_type, model_name)
        os.makedirs(model_dir, exist_ok=True)

        train_model(
            data_dir=data_dir,
            batch_size=16,
            learning_rate=learning_rate,
            num_epochs=100,
            device=device,
            save_path=os.path.join(model_dir, "best_model.pth"),
            target_metric=metric,
            until_convergence=True,
            patience=10,
            dropout_rate=dropout_rate,
            num_fc_units=num_fc_units,
            conv_filters=conv_filters,
            model_variant=(
                "multi" if model_type == "best_model"
                else "iqdcnn_multi" if model_type == "IQDCNN"
                else "efficientnet_multi" if model_type == "EfficientNetIQA"
                else ValueError(f"Unsupported model_type for metrics: {model_type}")
            ),
            loss_fn=loss_fn,
            optimizer=optimizer
        )
    else:
        raise ValueError(f"Invalid metric config: {metric}")