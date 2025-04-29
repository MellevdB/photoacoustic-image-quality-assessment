import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dl_model.model_definition import PhotoacousticQualityNet
from dl_model.utils import PhotoacousticDataset, create_train_val_test_split

# 0) Define custom Anti-Bias L1 loss function
class AntiBiasL1Loss(nn.Module):
    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs(y_pred - y_true))  # Simple L1 loss for now

# Main training loop
def train_model(
    data_dir,
    batch_size=8,
    learning_rate=1e-3,
    num_epochs=10,
    device='cuda',
    save_path="best_model.pth",
    target_metric="CLIP-IQA"
):
    # 1) Create train and val datasets from combined CSVs
    train_data, val_data, _ = create_train_val_test_split(data_dir, target_metric=target_metric)

    # 2) Wrap them in DataLoader for batch processing
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # 3) Initialize the CNN model and move to device
    model = PhotoacousticQualityNet(in_channels=1).to(device)

    # 4) Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = AntiBiasL1Loss()

    best_val_loss = float('inf')  # for model checkpointing

    # 5) Epoch loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).unsqueeze(1).float()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)

        train_loss /= len(train_loader.dataset)

        # 6) Validation loop
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

        # 7) Logging and saving the best model
        print(f"[Epoch {epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Ensure save directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'in_channels': 1,
                'state_dict': model.state_dict(),
            }, save_path)
            print(f"✔ New best model saved to {save_path} with val_loss={val_loss:.4f}")

    return model

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Path to directory with CSV files")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save_path", type=str, default="best_model.pth")

    args = parser.parse_args()

    train_model(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        device=args.device,
        save_path=args.save_path
    )