import os
import glob
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from dl_model.inference import load_model_checkpoint  # uses your existing loader

# ------------- Config -------------
device = "cuda" if torch.cuda.is_available() else "cpu"
image_dir = "/home/mvanderbrugge/photoacoustic-image-quality-assessment/survey"
output_dir = os.path.join(image_dir, "predictions")
os.makedirs(output_dir, exist_ok=True)

# If your models were trained multi-output on these 5 IQA metrics:
metrics = ['SSIM', 'GMSD_norm', 'HAARPSI', 'S3IM', 'IWSSIM']
metric_name = "_".join(metrics)

# Map display names to your actual checkpoint locations
model_paths = {
    "PAQNet":          os.path.join("models", "best_model",        metric_name, "best_model.pth"),
    "IQDCNN":          os.path.join("models", "IQDCNN",            metric_name, "best_model.pth"),
    "EfficientNetIQA": os.path.join("models", "EfficientNetIQA",   metric_name, "best_model.pth"),
}

# Preprocessing (grayscale, resize 128×128, ToTensor in [0,1])
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),  # produces [1,H,W] for L-mode
])

# ------------- Dataset -------------
ALLOWED_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff")

class FolderImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        p = self.image_paths[idx]
        img = Image.open(p).convert("L")  # enforce grayscale like training
        tensor = self.transform(img)      # [1,128,128]
        return tensor, p

# Collect images
image_paths = sorted(
    [p for p in glob.glob(os.path.join(image_dir, "*")) if os.path.splitext(p)[1].lower() in ALLOWED_EXT]
)
if len(image_paths) == 0:
    raise FileNotFoundError(f"No images found in {image_dir} with extensions {ALLOWED_EXT}")

dataset = FolderImageDataset(image_paths, transform)
loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)

# ------------- Run inference for each model -------------
per_model_csvs = []
merged_df = pd.DataFrame({"image_path": image_paths})

for model_name, ckpt_path in model_paths.items():
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Missing checkpoint for {model_name}: {ckpt_path}")

    model = load_model_checkpoint(ckpt_path, device=device)
    model.eval()

    all_preds = []
    all_paths = []

    with torch.no_grad():
        for batch_imgs, batch_paths in loader:
            batch_imgs = batch_imgs.to(device)
            outputs = model(batch_imgs)              # shape [B, K] (K = 5 metrics) or [B,1]
            preds = outputs.detach().cpu().numpy()
            all_preds.append(preds)
            all_paths.extend(batch_paths)

    preds = np.concatenate(all_preds, axis=0)        # [N, K] or [N,1]

    # Build a tidy DataFrame for this model
    df = pd.DataFrame({"image_path": all_paths})

    if preds.ndim == 1:
        preds = preds.reshape(-1, 1)

    num_out = preds.shape[1]
    if num_out == len(metrics):
        for i, m in enumerate(metrics):
            df[f"pred_{model_name}_{m}"] = preds[:, i]
            # also add to merged_df
            merged_df[f"pred_{model_name}_{m}"] = preds[:, i]
    else:
        # Fallback: unknown number of outputs — name them generically
        for i in range(num_out):
            df[f"pred_{model_name}_metric_{i}"] = preds[:, i]
            merged_df[f"pred_{model_name}_metric_{i}"] = preds[:, i]

    # Save per-model CSV
    per_model_csv = os.path.join(output_dir, f"predictions_{model_name}.csv")
    df.to_csv(per_model_csv, index=False)
    per_model_csvs.append(per_model_csv)
    print(f"[✓] Saved: {per_model_csv}")

# Save merged CSV (all models side-by-side)
merged_csv = os.path.join(output_dir, "predictions_all_models.csv")
merged_df.to_csv(merged_csv, index=False)
print(f"[✓] Saved combined: {merged_csv}")

# (Optional) Pretty print quick summary
print("\nPreview of merged predictions:")
print(merged_df.head())