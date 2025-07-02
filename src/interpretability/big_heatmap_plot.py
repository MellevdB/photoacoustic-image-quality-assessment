import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from PIL import Image
import os
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from dl_model.inference import load_model_checkpoint

# ---------------------- Settings ----------------------
device = "cuda"
patch_size = 16
stride = 8
model_paths = {
    "best_model": "models/best_model/SSIM/best_model.pth",
    "IQDCNN": "models/IQDCNN/SSIM/best_model.pth",
    "EfficientNetIQA": "models/EfficientNetIQA/SSIM/best_model.pth",
}
datasets_order = ["Mice", "SWFD", "SCD_ms", "SCD_vc", "PhantomsEFA"]
image_paths = {
    "Mice": "results/mice/images_used/full_recon_all_slice_74.png",
    "SWFD": "results/SWFD/images_used/sc,ss128_BP_slice_89.png",
    "SCD_ms": "results/SCD/images_used/ms,ss128_BP_slice_62.png",
    "SCD_vc": "results/SCD/images_used/vc,ss128_BP_slice_44.png",
    "PhantomsEFA": "data/VARIED SPLIT V3 CURRENT/scene_5003/BVPhantom_Rf_102622_020539 PA4.webp"
}
model_display = {
    "original": "Original",
    "best_model": "PhotoacousticQualityNet",
    "IQDCNN": "IQDCNN",
    "EfficientNetIQA": "EfficientNetIQA"
}

# Resize + normalize
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

def generate_heatmap(model, image_tensor):
    model.eval()
    with torch.no_grad():
        baseline = model(image_tensor).item()

    _, _, H, W = image_tensor.shape
    heatmap = np.zeros((H, W))
    counts = np.zeros((H, W))

    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            occluded = image_tensor.clone()
            occluded[:, :, i:i+patch_size, j:j+patch_size] = 0.0
            with torch.no_grad():
                pred = model(occluded).item()
            score_drop = baseline - pred
            heatmap[i:i+patch_size, j:j+patch_size] += score_drop
            counts[i:i+patch_size, j:j+patch_size] += 1

    counts[counts == 0] = 1
    return heatmap / counts

# Load models
models = {name: load_model_checkpoint(path, device=device) for name, path in model_paths.items()}

# Generate all heatmaps and collect
all_data = []
vmins, vmaxs = [], []

for dataset in datasets_order:
    row_data = []
    row_min, row_max = None, None

    img = Image.open(image_paths[dataset]).convert("L").resize((128, 128))
    original_np = np.array(img) / 255.0
    row_data.append(original_np)

    input_tensor = transform(img).unsqueeze(0).to(device)
    for model_key in model_paths:
        heatmap = generate_heatmap(models[model_key], input_tensor)
        row_data.append(heatmap)
        row_min = np.min(heatmap) if row_min is None else min(row_min, np.min(heatmap))
        row_max = np.max(heatmap) if row_max is None else max(row_max, np.max(heatmap))

    all_data.append(row_data)
    vmins.append(row_min)
    vmaxs.append(row_max)

fig = plt.figure(figsize=(18, 28))  # ⬅️ reduced width for better tick visibility
outer = gridspec.GridSpec(len(datasets_order) + 1, 1,
                          height_ratios=[1]*len(datasets_order) + [0.1],
                          hspace=0.15)

# Plot rows (same as before)
for row_idx, (row_data, dataset_name) in enumerate(zip(all_data, datasets_order)):
    inner = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[row_idx], wspace=0.02)
    vmin = vmins[row_idx]
    vmax = vmaxs[row_idx]
    row_axes = []

    for col_idx in range(4):
        ax = plt.Subplot(fig, inner[col_idx])
        ax.set_xticks([])
        ax.set_yticks([])

        if col_idx == 0:
            ax.imshow(row_data[0], cmap="gray")
            ax.text(-0.06, 0.5, dataset_name, va="center", ha="right",
                    fontsize=28, transform=ax.transAxes,
                    rotation=90, fontweight='bold')
        else:
            ax.imshow(row_data[0], cmap="gray")
            ax.imshow(row_data[col_idx], cmap="jet", alpha=0.5, vmin=vmin, vmax=vmax)

        fig.add_subplot(ax)
        row_axes.append(ax)

    # Right-side vertical colorbar (no overlap)
    last_ax = row_axes[-1]
    divider = make_axes_locatable(last_ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)  # 5% width and 0.05 padding
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label("")

# Add model names under last row
inner_bottom = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer[-1], wspace=0.02)
for col_idx, model_key in enumerate(["original"] + list(model_paths.keys())):
    ax = plt.Subplot(fig, inner_bottom[col_idx])
    ax.axis("off")
    ax.text(0.5, 0.5, model_display[model_key],
            ha="center", va="center", fontsize=28, fontweight='bold',
            transform=ax.transAxes)
    fig.add_subplot(ax)

# Tighter margin layout (more room for tick labels)
fig.subplots_adjust(left=0.03, right=0.965, top=0.98, bottom=0.03)

output_path = "results/visualize/Heatmaps/combined_occlusion_heatmap_grid_per_row_colorbars_final.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300)
plt.close()
print(f"[✓] Saved: {output_path}")