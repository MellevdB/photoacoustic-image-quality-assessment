# occlusion/heatmap.py
import os
from pathlib import Path
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import transforms
from dl_model.inference import load_model_checkpoint

# ---------------------- Settings ----------------------
device = "cuda"
patch_size = 16
stride = 8

# Models
model_paths = {
    "best_model": "models/best_model_thesis/SSIM/best_model.pth",
    "IQDCNN": "models/IQDCNN_thesis/SSIM/best_model.pth",
    "EfficientNetIQA": "models/EfficientNetIQA_thesis/SSIM/best_model.pth",
}
model_display = {
    "original": "Original",
    "best_model": "PAQNet",
    "IQDCNN": "IQDCNN",
    "EfficientNetIQA": "EfficientNetIQA",
}

# Font sizes
tick_fontsize = 15
label_fontsize = 24
subplot_label_fontsize = 30
model_label_fontsize = 36
colorbar_number_fontsize = 20

# Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ---------------------- Dataset Images ----------------------
IMROOT = Path("occlusion/images")
DATASETS = {
    "SCD_ms": [
        ("sparse32", IMROOT / "SCD_multisegment_ms,ss32_BP_slice81.png"),
        ("sparse64", IMROOT / "SCD_multisegment_ms,ss64_BP_slice81.png"),
        ("sparse128", IMROOT / "SCD_multisegment_ms,ss128_BP_slice81.png"),
    ],
    "SWFD_ms": [
        ("sparse32", IMROOT / "SWFD_multisegment_ms,ss32_BP_slice0082.png"),
        ("sparse64", IMROOT / "SWFD_multisegment_ms,ss64_BP_slice0082.png"),
        ("sparse128", IMROOT / "SWFD_multisegment_ms,ss128_BP_slice0082.png"),
    ],
    "SWFD_sc": [
        ("sparse32", IMROOT / "SWFD_semicircle_sc,ss32_BP_slice0031.png"),
        ("sparse64", IMROOT / "SWFD_semicircle_sc,ss64_BP_slice0031.png"),
        ("sparse128", IMROOT / "SWFD_semicircle_sc,ss128_BP_slice0031.png"),
    ],
    "Mice": [
        ("k=4", IMROOT / "mice_image_113_4.png"),
        ("k=32", IMROOT / "mice_image_113_32.png"),
        ("k=256", IMROOT / "mice_image_113_256.png"),
    ],
    "Phantom": [
        ("k=8", IMROOT / "phantom_image_212_8.png"),
        ("k=32", IMROOT / "phantom_image_212_32.png"),
        ("k=128", IMROOT / "phantom_image_212_128.png"),
    ],
}

# ---------------------- Metrics helpers ----------------------
def _normalize01(x: np.ndarray) -> np.ndarray:
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + 1e-8)

def _compare_maps(heatmap: np.ndarray, gradcam: np.ndarray):
    H = _normalize01(heatmap)
    G = _normalize01(gradcam)
    Hf, Gf = H.ravel(), G.ravel()

    l2 = float(np.sqrt(np.mean((Hf - Gf) ** 2)))
    H_std = float(Hf.std()); G_std = float(Gf.std())
    if H_std < 1e-12 or G_std < 1e-12:
        corr = np.nan
    else:
        corr = float(np.corrcoef(Hf, Gf)[0, 1])
    return l2, corr

def _append_metrics_row(dataset_key, row_label, model_key, l2, corr):
    metrics_dir = Path("occlusion/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    csv_path = metrics_dir / "heatmap_gradcam_similarity.csv"
    header_needed = not csv_path.exists()
    with open(csv_path, "a") as f:
        if header_needed:
            f.write("dataset,row_label,model,l2_distance,pearson_corr\n")
        f.write(f"{dataset_key},{row_label},{model_key},{l2:.6f},{corr if not np.isnan(corr) else 'nan'}\n")

# ---------------------- Core Functions ----------------------
def generate_heatmap(model, image_tensor, patch_size=16, stride=8):
    model.eval()
    with torch.no_grad():
        baseline = model(image_tensor).item()

    _, _, H, W = image_tensor.shape
    heatmap = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)

    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            occluded = image_tensor.clone()
            occluded[:, :, i:i+patch_size, j:j+patch_size] = 0.0
            with torch.no_grad():
                pred = model(occluded).item()
            score_drop = baseline - pred
            heatmap[i:i+patch_size, j:j+patch_size] += score_drop
            counts[i:i+patch_size, j:j+patch_size] += 1.0

    counts[counts == 0] = 1.0
    return heatmap / counts

def prepare_row_data(img_path, models_by_key, dataset_key, row_label):
    img = Image.open(img_path).convert("L").resize((128, 128))
    original_np = np.array(img, dtype=np.float32) / 255.0
    input_tensor = transform(img).unsqueeze(0).to(device)

    row_data = [original_np]
    row_min, row_max = None, None

    for model_key in model_paths:
        hm = generate_heatmap(models_by_key[model_key], input_tensor, patch_size, stride)
        row_data.append(hm)
        row_min = np.min(hm) if row_min is None else min(row_min, float(np.min(hm)))
        row_max = np.max(hm) if row_max is None else max(row_max, float(np.max(hm)))

        # --- Save raw heatmap map for metrics cross-compare ---
        out_npy = Path(f"occlusion/maps/heatmap/{dataset_key}")
        out_npy.mkdir(parents=True, exist_ok=True)
        np.save(out_npy / f"{row_label}_{model_key}.npy", hm.astype(np.float32))

        # If corresponding gradcam exists, compute metrics now
        grad_npy = Path(f"occlusion/maps/gradcam/{dataset_key}/{row_label}_{model_key}.npy")
        if grad_npy.exists():
            grad_map = np.load(grad_npy)
            l2, corr = _compare_maps(hm, grad_map)
            _append_metrics_row(dataset_key, row_label, model_key, l2, corr)

    return row_data, row_min, row_max

def plot_dataset_rows(dataset_key, rows, models_by_key):
    all_rows, vmins, vmaxs = [], [], []
    for row_label, img_path in rows:
        row_data, rmin, rmax = prepare_row_data(img_path, models_by_key, dataset_key, row_label)
        all_rows.append((row_label, row_data))
        vmins.append(rmin)
        vmaxs.append(rmax)

    global_vmin, global_vmax = min(vmins), max(vmaxs)
    global_vcenter = 0.5 * (global_vmin + global_vmax)

    fig = plt.figure(figsize=(18, 18))
    outer = gridspec.GridSpec(len(rows) + 1, 1, height_ratios=[0.12] + [1] * len(rows), hspace=0.25)

    # Header
    inner_top = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[0], wspace=0.02)
    for col_idx, key in enumerate(["original"] + list(model_paths.keys())):
        ax = plt.Subplot(fig, inner_top[col_idx])
        ax.axis("off")
        ax.text(0.5, 0.5, model_display[key], ha="center", va="center",
                fontsize=model_label_fontsize, fontweight='bold', transform=ax.transAxes)
        fig.add_subplot(ax)

    last_row_axes = None
    for r, (row_label, row_data) in enumerate(all_rows):
        inner = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[r + 1], wspace=0.02)
        row_letter = chr(65 + r)
        row_axes = []

        for c in range(4):
            ax = plt.Subplot(fig, inner[c])
            ax.set_xticks([]); ax.set_yticks([])

            if c == 0:
                ax.imshow(row_data[0], cmap="gray")
                ax.text(0.02, 0.5, row_label, va="center", ha="left",
                        fontsize=label_fontsize, transform=ax.transAxes,
                        rotation=90, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.25", facecolor='white', alpha=0.7, linewidth=0))
            else:
                ax.imshow(row_data[0], cmap="gray")
                ax.imshow(row_data[c], cmap="jet", alpha=0.5, vmin=global_vmin, vmax=global_vmax)

            ax.text(0.05, 0.95, f"{row_letter}{c + 1}", ha="left", va="top",
                    fontsize=subplot_label_fontsize, fontweight='bold',
                    transform=ax.transAxes, color='white',
                    bbox=dict(boxstyle="round,pad=0.30", facecolor='black', alpha=0.7, linewidth=0))

            fig.add_subplot(ax)
            row_axes.append(ax)

        if r == len(all_rows) - 1:
            last_row_axes = row_axes

    fig.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.12)

    # Colorbar
    fig.canvas.draw()
    left_ax = last_row_axes[0].get_position(fig)
    right_ax = last_row_axes[-1].get_position(fig)
    cb_left, cb_right = left_ax.x0, right_ax.x1
    cb_width, cb_height, cb_bottom = cb_right - cb_left, 0.035, left_ax.y0 - 0.06

    cax = fig.add_axes([cb_left, cb_bottom, cb_width, cb_height])
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=global_vmin, vmax=global_vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')

    tick_offset = (global_vmax - global_vmin) * 0.10
    adjusted_vmin = global_vmin + tick_offset
    cbar.set_ticks([adjusted_vmin, global_vcenter, global_vmax])
    cbar.ax.set_xticklabels([f"{global_vmin:.3f}", f"{global_vcenter:.3f}", f"{global_vmax:.3f}"],
                            fontsize=colorbar_number_fontsize)
    cbar.ax.tick_params(axis='x', which='major', pad=15, labelsize=colorbar_number_fontsize)
    cbar.set_label("Δ Score", fontsize=label_fontsize)

    for spine in cbar.ax.spines.values():
        spine.set_visible(False)

    # Save figures
    out_dir = Path("occlusion/heatmaps")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"heatmap_{dataset_key}.png"
    fig.savefig(out_path.as_posix(), dpi=300)
    plt.close(fig)
    print(f"[✓] Saved: {out_path}")

# ---------------------- Load Models ----------------------
models = {k: load_model_checkpoint(p, device=device) for k, p in model_paths.items()}

# ---------------------- Run for All Datasets ----------------------
if __name__ == "__main__":
    for key, rows in DATASETS.items():
        for _, p in rows:
            if not Path(p).exists():
                raise FileNotFoundError(f"Missing image for {key}: {p}")
        plot_dataset_rows(key, rows, models)