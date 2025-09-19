import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from collections import defaultdict
from torchsummary import summary
from dl_model.model_definition import PhotoacousticQualityNet, PhotoacousticQualityNetBN, PhotoacousticQualityNetMulti, IQDCNN, EfficientNetIQA, IQDCNNMulti, EfficientNetIQAMulti
from dl_model.utils import create_train_val_test_split

class AntiBiasL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        y_pred = y_pred.squeeze()
        y_true = y_true.squeeze()
        grade_buckets = defaultdict(list)
        for i, g in enumerate(y_true.tolist()):
            grade_buckets[g].append(i)
        losses = []
        for g, idxs in grade_buckets.items():
            idxs = torch.tensor(idxs, device=y_true.device)
            losses.append(torch.mean(torch.abs(y_pred[idxs] - y_true[idxs])))
        return torch.mean(torch.stack(losses))

def get_loss_function(loss_fn_type):
    if loss_fn_type == "l1":
        return nn.L1Loss()
    elif loss_fn_type == "mse":
        return nn.MSELoss()
    elif loss_fn_type == "huber":
        return nn.HuberLoss()
    elif loss_fn_type == "antibias":
        return AntiBiasL1Loss()
    else:
        raise ValueError(f"Unknown loss function: {loss_fn_type}")

def get_optimizer(params, optimizer_type, lr):
    if optimizer_type == "adam":
        return optim.Adam(params, lr=lr)
    elif optimizer_type == "adamw":
        return optim.AdamW(params, lr=lr)
    elif optimizer_type == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

def train_model(
    data_dir,
    batch_size,
    learning_rate,
    num_epochs,
    device,
    save_path,
    target_metric,
    until_convergence,
    patience,
    dropout_rate,
    num_fc_units,
    conv_filters,
    model_variant,
    loss_fn,
    optimizer
):
    train_data, val_data, _ = create_train_val_test_split(data_dir, target_metric=target_metric)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    print(f"Target metric: {target_metric}")

    print("Computing target value ranges...")
    if isinstance(target_metric, list):
        train_targets = torch.stack([sample[1] for sample in train_data])  # shape [N, num_metrics]
        mins = train_targets.min(dim=0).values
        maxs = train_targets.max(dim=0).values
        for i, m in enumerate(target_metric):
            print(f"  {m}: min={mins[i]:.4f}, max={maxs[i]:.4f}")
    else:
        train_targets = torch.tensor([sample[1].item() for sample in train_data])
        print(f"Target range → min: {train_targets.min():.4f}, max: {train_targets.max():.4f}")

    num_outputs = len(target_metric) if isinstance(target_metric, list) else 1

    if model_variant == "dropout":
        print("Using PhotoacousticQualityNet")
        model = PhotoacousticQualityNet(
            in_channels=1,
            conv_filters=conv_filters,
            dropout_rate=dropout_rate,
            num_fc_units=num_fc_units
        ).to(device)

    elif model_variant == "batchnorm":
        print("Using PhotoacousticQualityNetBN")
        model = PhotoacousticQualityNetBN(
            in_channels=1,
            conv_filters=conv_filters,
            num_fc_units=num_fc_units
        ).to(device)

    elif model_variant == "multi":
        print("Using PhotoacousticQualityNetMulti")
        num_outputs = len(target_metric)
        model = PhotoacousticQualityNetMulti(
            in_channels=1,
            conv_filters=conv_filters,
            dropout_rate=dropout_rate,
            num_fc_units=num_fc_units,
            num_outputs=num_outputs
        ).to(device)

    elif model_variant == "iqdcnn":
        print("Using IQDCNN")
        model = IQDCNN(
            in_channels=1,
            conv_filters=conv_filters,
            dropout_rate=dropout_rate,
            num_fc_units=num_fc_units
        ).to(device)

    elif model_variant == "efficientnet":
        print("Using EfficientNetIQA")
        model = EfficientNetIQA(pretrained=True).to(device)

    elif model_variant == "iqdcnn_multi":
        print("Using IQDCNNMulti")
        model = IQDCNNMulti(
            in_channels=1,
            conv_filters=conv_filters,
            dropout_rate=dropout_rate,
            num_fc_units=num_fc_units,
            num_outputs=num_outputs
        ).to(device)

    elif model_variant == "efficientnet_multi":
        print('Using EfficientNetIQAMulti')
        model = EfficientNetIQAMulti(pretrained=True, num_outputs=num_outputs).to(device)

    else:
        raise ValueError(f"Invalid model_variant: {model_variant}")

    summary(model, input_size=(1, 128, 128))

    criterion = get_loss_function(loss_fn)
    optimizer = get_optimizer(model.parameters(), optimizer, learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    epoch = 0
    loss_log = []

    while True:
        if epoch == 0:
            print("Entering first epoch of training loop")
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            images, labels = batch[0], batch[1]
            images = images.to(device)
            if num_outputs > 1:
                labels = labels.to(device).float()  # shape: [B, 3]
            else:
                labels = labels.to(device).unsqueeze(1).float()  # shape: [B, 1]
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch[0], batch[1]
                images = images.to(device)
                if model_variant == "multi":
                    labels = labels.to(device).float()  # shape: [B, 3]
                else:
                    labels = labels.to(device).unsqueeze(1).float()  # shape: [B, 1]
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        loss_log.append((epoch + 1, train_loss, val_loss))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'in_channels': 1,
                'state_dict': model.state_dict(),
                'conv_filters': conv_filters,
                'dropout_rate': dropout_rate,
                'num_fc_units': num_fc_units,
                'model_variant': model_variant,
                'num_outputs': num_outputs,
            }, save_path)
            print(f"New best model saved to {save_path} with val_loss={val_loss:.4f}")
        else:
            epochs_no_improve += 1

        epoch += 1
        if not until_convergence and epoch >= num_epochs:
            break
        if until_convergence and epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} no-improve epochs.")
            break

    loss_df = pd.DataFrame(loss_log, columns=["epoch", "train_loss", "val_loss"])
    loss_csv_path = os.path.join(os.path.dirname(save_path), "train_val_loss.csv")
    loss_df.to_csv(loss_csv_path, index=False)
    print(f"Saved training log to {loss_csv_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_path", type=str, default="best_model.pth")
    parser.add_argument("--target_metric", type=str, required=True)
    parser.add_argument("--until_convergence", action="store_true")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--num_fc_units", type=int, default=128)
    parser.add_argument("--conv_filters", type=int, nargs=4, default=[16, 32, 64, 128])
    parser.add_argument(
    "--model_variant",
    type=str,
    choices=[
        "dropout", "batchnorm", "multi",
        "iqdcnn", "iqdcnn_multi",
        "efficientnet", "efficientnet_multi"
    ],
    default="dropout"
    )
    parser.add_argument("--loss_fn", type=str, choices=["l1", "mse", "huber", "antibias"], default="antibias")
    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw", "sgd"], default="adam")
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        device=args.device,
        save_path=args.save_path,
        target_metric=args.target_metric,
        until_convergence=args.until_convergence,
        patience=args.patience,
        dropout_rate=args.dropout_rate,
        num_fc_units=args.num_fc_units,
        conv_filters=args.conv_filters,
        model_variant=args.model_variant,
        loss_fn=args.loss_fn,
        optimizer=args.optimizer
    )