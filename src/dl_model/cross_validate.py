# cross_validate.py

# cross_validate.py
import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from dl_model.model_definition import PhotoacousticQualityNet, PhotoacousticQualityNetBN
from dl_model.utils import PhotoacousticDatasetFromDataFrame
from dl_model.train import AntiBiasL1Loss
import torch.nn as nn
import torch.optim as optim
import ast

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
        raise ValueError(f"Unknown loss: {loss_fn_type}")

def get_optimizer(params, optimizer_type, lr):
    if optimizer_type == "adam":
        return optim.Adam(params, lr=lr)
    elif optimizer_type == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9)
    elif optimizer_type == "adamw":
        return optim.AdamW(params, lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

def cross_validate(args):
    df = pd.read_csv(args.csv_path)
    df = df.dropna(subset=[args.metric])

    kf = KFold(n_splits=args.num_folds, shuffle=True, random_state=42)
    losses = []

    conv_filters = ast.literal_eval(args.conv_filters)

    for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
        print(f"\n[Fold {fold + 1}]")

        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)

        train_data = PhotoacousticDatasetFromDataFrame(train_df, target_metric=args.metric)
        val_data = PhotoacousticDatasetFromDataFrame(val_df, target_metric=args.metric)

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

        # Select model
        if args.model_type == "batchnorm":
            model = PhotoacousticQualityNetBN(in_channels=1, conv_filters=conv_filters, num_fc_units=args.fc_units).to(args.device)
        elif args.model_type == "dropout":
            model = PhotoacousticQualityNet(in_channels=1, conv_filters=conv_filters, num_fc_units=args.fc_units, dropout_rate=args.dropout).to(args.device)
        else:
            raise ValueError("Invalid model type")

        optimizer = get_optimizer(model.parameters(), args.optimizer, args.lr)
        criterion = get_loss_function(args.loss_fn)

        for epoch in range(args.num_epochs):
            model.train()
            for images, labels in train_loader:
                images = images.to(args.device)
                labels = labels.to(args.device).unsqueeze(1).float()
                optimizer.zero_grad()
                loss = criterion(model(images), labels)
                loss.backward()
                optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(args.device)
                labels = labels.to(args.device).unsqueeze(1).float()
                outputs = model(images)
                loss = nn.L1Loss()(outputs, labels)
                val_loss += loss.item() * images.size(0)

        val_loss /= len(val_loader.dataset)
        print(f"Val loss (L1): {val_loss:.4f}")
        losses.append(val_loss)

    avg_loss = np.mean(losses)

    out_dir = os.path.join("results", "eval_model", "cross_validate", args.model_type, args.metric)
    os.makedirs(out_dir, exist_ok=True)

    name = f"{args.model_type}_{args.loss_fn}_{args.optimizer}_bs{args.batch_size}_fc{args.fc_units}_drop{args.dropout}"
    name += "_conv" + "_".join(str(x) for x in conv_filters)
    result_path = os.path.join(out_dir, f"summary_{name}.txt")

    with open(result_path, "w") as f:
        f.write(f"Metric         : {args.metric}\n")
        f.write(f"Model          : {args.model_type}\n")
        f.write(f"Loss           : {args.loss_fn}\n")
        f.write(f"Optimizer      : {args.optimizer}\n")
        f.write(f"Dropout        : {args.dropout}\n")
        f.write(f"FC Units       : {args.fc_units}\n")
        f.write(f"Conv Filters   : {conv_filters}\n")
        f.write(f"Batch Size     : {args.batch_size}\n")
        f.write(f"LR             : {args.lr}\n")
        f.write(f"Epochs         : {args.num_epochs}\n")
        f.write(f"Folds          : {args.num_folds}\n")
        f.write(f"Mean Val Loss  : {avg_loss:.6f}\n")
        f.write(f"Fold losses    : {losses}\n")

    print(f"Saved summary to {result_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--metric", type=str, required=True)
    parser.add_argument("--model_type", type=str, choices=["dropout", "batchnorm"], default="dropout")
    parser.add_argument("--loss_fn", type=str, choices=["l1", "mse", "huber", "antibias"], default="l1")
    parser.add_argument("--optimizer", type=str, choices=["adam", "adamw", "sgd"], default="adam")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--fc_units", type=int, default=128)
    parser.add_argument("--conv_filters", type=str, default="[16, 32, 64, 128]")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--num_folds", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    cross_validate(args)