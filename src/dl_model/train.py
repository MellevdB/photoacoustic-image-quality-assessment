import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dl_model.model_definition import PhotoacousticQualityNet
from dl_model.utils import create_train_val_test_split
import pandas as pd
from collections import defaultdict

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

# Main training loop with optional early stopping

def train_model(
    data_dir,
    batch_size=8,
    learning_rate=5e-5,
    num_epochs=10,
    device='cuda',
    save_path="best_model.pth",
    target_metric="CLIP-IQA",
    until_convergence=False,
    patience=10,
    dropout_rate=0.3,
    num_fc_units=128,
):
    train_data, val_data, _ = create_train_val_test_split(data_dir, target_metric=target_metric)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    train_targets = [label.item() for _, label in train_data]
    print(f" Target range → min: {min(train_targets):.4f}, max: {max(train_targets):.4f}")

    model = PhotoacousticQualityNet(in_channels=1, dropout_rate=dropout_rate, num_fc_units=num_fc_units).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5, verbose=True)
    criterion = AntiBiasL1Loss()

    best_val_loss = float('inf')
    loss_log = []
    epochs_no_improve = 0
    epoch = 0

    while True:
        model.train()
        train_loss = 0.0
        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1).float()
            optimizer.zero_grad()
            outputs = model(images)

            if epoch == 0 and batch_idx == 0:
                print(f"[Debug] Sample predictions: {outputs[:5].squeeze().detach().cpu().numpy()}")
                print(f"[Debug] Corresponding targets: {labels[:5].squeeze().detach().cpu().numpy()}")

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).unsqueeze(1).float()
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
    print(f"Saved training loss log to {loss_csv_path}")
    return model

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_path", type=str, default="best_model.pth")
    parser.add_argument("--until_convergence", action="store_true")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--num_fc_units", type=int, default=128)
    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        device=args.device,
        save_path=args.save_path,
        until_convergence=args.until_convergence,
        patience=args.patience,
        dropout_rate=args.dropout_rate,
        num_fc_units=args.num_fc_units
    )