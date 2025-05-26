import os
import itertools
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold
from dl_model.model_definition import PhotoacousticQualityNet
from dl_model.utils import PhotoacousticDatasetFromDataFrame
from dl_model.train import AntiBiasL1Loss
import torch.nn as nn
import torch.optim as optim


def cross_validate_model(
    dataframe,
    metric,
    dropout_rates=[0.0, 0.3, 0.5, 0.7],
    learning_rates=[1e-4, 5e-5],
    fc_units=[128],
    batch_size=16,
    num_epochs=10,
    num_folds=3,
    device="cuda",
):
    results = []
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

    param_grid = list(itertools.product(dropout_rates, learning_rates, fc_units))

    for dropout, lr, fc_unit in param_grid:
        fold_losses = []
        print(f"\n Testing config: dropout={dropout}, lr={lr}, fc_units={fc_unit}")

        for fold, (train_idx, val_idx) in enumerate(kf.split(dataframe)):
            train_df = dataframe.iloc[train_idx].reset_index(drop=True)
            val_df = dataframe.iloc[val_idx].reset_index(drop=True)

            train_data = PhotoacousticDatasetFromDataFrame(train_df, target_metric=metric)
            val_data = PhotoacousticDatasetFromDataFrame(val_df, target_metric=metric)

            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

            model = PhotoacousticQualityNet(in_channels=1, num_fc_units=fc_unit, dropout_rate=dropout).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = AntiBiasL1Loss()

            for epoch in range(num_epochs):
                model.train()
                for images, labels in train_loader:
                    images = images.to(device)
                    labels = labels.to(device).unsqueeze(1).float()
                    optimizer.zero_grad()
                    loss = criterion(model(images), labels)
                    loss.backward()
                    optimizer.step()

            # Evaluate after last epoch
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for images, labels in val_loader:
                    images = images.to(device)
                    labels = labels.to(device).unsqueeze(1).float()
                    outputs = model(images)
                    loss = nn.L1Loss()(outputs, labels)
                    val_loss += loss.item() * images.size(0)

            val_loss /= len(val_loader.dataset)
            print(f"   Fold {fold + 1} val loss: {val_loss:.4f}")
            fold_losses.append(val_loss)

        avg_loss = np.mean(fold_losses)
        results.append({
            "dropout": dropout,
            "learning_rate": lr,
            "fc_units": fc_unit,
            "avg_val_loss": avg_loss
        })

    result_df = pd.DataFrame(results).sort_values("avg_val_loss")
    result_df.to_csv(f"cv_results_{metric}.csv", index=False)
    print("\n Cross-validation finished. Top configs:")
    print(result_df.head())
    return result_df


if __name__ == "__main__":
    import argparse
    from glob import glob

    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--metric", type=str, default="CLIP-IQA")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    df = df.dropna(subset=[args.metric])  # Drop rows where target is NaN

    cross_validate_model(
        dataframe=df,
        metric=args.metric,
        device=args.device
    )
