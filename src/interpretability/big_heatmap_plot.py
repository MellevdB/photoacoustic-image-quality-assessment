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
    "best_model": "PAQNet",
    "IQDCNN": "IQDCNN",
    "EfficientNetIQA": "EfficientNetIQA"
}

# === Font size settings (matching plot_metric_scores.py wide format) ===
tick_fontsize = 15
label_fontsize = 24  # reduced from 30 to prevent cutoff (dataset names)
subplot_label_fontsize = 30  # increased from 24
model_label_fontsize = 36  # increased from 28 (model names)
colorbar_number_fontsize = 20  # increased from 13 (colorbar numbers)

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

# Calculate global min/max for consistent color scaling across all images
global_vmin = min(vmins)
global_vmax = max(vmaxs)
global_vcenter = (global_vmin + global_vmax) / 2

fig = plt.figure(figsize=(18, 26))  # Reduced height since colorbar is now absolutely positioned
outer = gridspec.GridSpec(len(datasets_order) + 1, 1,  # +1 for model names only
                          height_ratios=[0.1] + [1]*len(datasets_order),  # model names + images
                          hspace=0.25)  # reduced spacing since no colorbar row

# Add model names at the top first
inner_top = gridspec.GridSpecFromSubplotSpec(
    1, 4, subplot_spec=outer[0], wspace=0.02  # Removed colorbar slot, now 4 columns
)
for col_idx, model_key in enumerate(["original"] + list(model_paths.keys())):
    ax = plt.Subplot(fig, inner_top[col_idx])
    ax.axis("off")
    ax.text(0.5, 0.5, model_display[model_key],
            ha="center", va="center", fontsize=model_label_fontsize, fontweight='bold',
            transform=ax.transAxes)
    fig.add_subplot(ax)

# Plot rows with A1-A4, B1-B4 labels
for row_idx, (row_data, dataset_name) in enumerate(zip(all_data, datasets_order)):
    # Now 4 columns (no colorbar slot needed)
    inner = gridspec.GridSpecFromSubplotSpec(
        1, 4,  # 4 images only
        subplot_spec=outer[row_idx + 1],
        wspace=0.02
    )
    
    # Generate row label (A1-A4, B1-B4, etc.)
    row_letter = chr(65 + row_idx)  # A, B, C, D, E
    
    for col_idx in range(4):
        ax = plt.Subplot(fig, inner[col_idx])
        ax.set_xticks([])
        ax.set_yticks([])

        if col_idx == 0:
            # Original image + dataset name on the left
            ax.imshow(row_data[0], cmap="gray")
            ax.text(-0.08, 0.5, dataset_name, va="center", ha="right",
                    fontsize=label_fontsize, transform=ax.transAxes,
                    rotation=90, fontweight='bold')
        else:
            # Heatmap overlay with global normalization
            ax.imshow(row_data[0], cmap="gray")
            ax.imshow(row_data[col_idx], cmap="jet", alpha=0.5, 
                     vmin=global_vmin, vmax=global_vmax)
        
        # Add A1, A2, A3, A4 labels inside the image (top-left corner)
        ax.text(0.05, 0.95, f"{row_letter}{col_idx + 1}", 
                ha="left", va="top", fontsize=subplot_label_fontsize, 
                fontweight='bold', transform=ax.transAxes, color='white',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.7))

        fig.add_subplot(ax)

# Add colorbar row at the bottom - create a colorbar that spans only the width of the 4 images
# Calculate the position to align with the image grid
left_pos = 0.03  # left margin
right_pos = 0.97  # right margin
image_width = (right_pos - left_pos) / 4  # width of each image column

# Create colorbar axes that spans exactly the width of the 4 images
cax = fig.add_axes([left_pos, 0.08, image_width * 4, 0.04])  # [left, bottom, width, height] - increased height for better visibility

# Create the colorbar
sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=global_vmin, vmax=global_vmax))
sm.set_array([])
cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')

# Configure colorbar ticks and labels
tick_offset = (global_vmax - global_vmin) * 0.10
adjusted_vmin = global_vmin + tick_offset
cbar.set_ticks([adjusted_vmin, global_vcenter, global_vmax])
cbar.ax.set_xticklabels([f"{global_vmin:.3f}", f"{global_vcenter:.3f}", f"{global_vmax:.3f}"], 
                        fontsize=colorbar_number_fontsize)
cbar.ax.tick_params(axis='x', which='major', pad=15, labelsize=colorbar_number_fontsize)
cbar.set_label("Δ Score", fontsize=label_fontsize)

# Remove colorbar border but keep ticks visible
for spine in cbar.ax.spines.values():
    spine.set_visible(False)

# Model names are now at the top, so remove this section

# Adjusted margin layout
fig.subplots_adjust(left=0.03, right=0.97, top=0.98, bottom=0.08)

output_path = "results/visualize/Heatmaps/combined_occlusion_heatmap_grid_with_labels_and_global_colorbar.png"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300)
plt.close()
print(f" Saved: {output_path}")

# ============================================================
# NEW FIGURE: SCD_ms — three rows (ss32, ss64, ss128), same slice
# ============================================================

# Row config (label, path)
scd_ms_rows = [
    ("SCD_ms_ss32", "results/SCD/images_used/ms,ss32_BP_slice_81.png"),
    ("SCD_ms_ss64", "results/SCD/images_used/ms,ss64_BP_slice_81.png"),
    ("SCD_ms_ss128", "results/SCD/images_used/ms,ss128_BP_slice_81.png"),
]

# Collect data per row (original + heatmaps for each model) and vmin/vmax per row
all_data_scd = []
vmins_scd, vmaxs_scd = [], []

for row_label, row_img_path in scd_ms_rows:
    row_data = []
    row_min, row_max = None, None

    img = Image.open(row_img_path).convert("L").resize((128, 128))
    original_np = np.array(img) / 255.0
    row_data.append(original_np)

    input_tensor = transform(img).unsqueeze(0).to(device)
    for model_key in model_paths:
        heatmap = generate_heatmap(models[model_key], input_tensor)
        row_data.append(heatmap)
        row_min = np.min(heatmap) if row_min is None else min(row_min, np.min(heatmap))
        row_max = np.max(heatmap) if row_max is None else max(row_max, np.max(heatmap))

    all_data_scd.append((row_label, row_data))
    vmins_scd.append(row_min)
    vmaxs_scd.append(row_max)

# Calculate global min/max for consistent color scaling across all SCD_ms images
global_vmin_scd = min(vmins_scd)
global_vmax_scd = max(vmaxs_scd)
global_vcenter_scd = (global_vmin_scd + global_vmax_scd) / 2

# Figure: same layout, but 3 rows (+ 1 header row only)
fig2 = plt.figure(figsize=(18, 18))
outer2 = gridspec.GridSpec(len(scd_ms_rows) + 1, 1,
                           height_ratios=[0.12] + [1]*len(scd_ms_rows),
                           hspace=0.25)

# Header row with model names (Original + models)
inner_top2 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer2[0], wspace=0.02)
for col_idx, model_key in enumerate(["original"] + list(model_paths.keys())):
    ax = plt.Subplot(fig2, inner_top2[col_idx])
    ax.axis("off")
    ax.text(0.5, 0.5, model_display[model_key],
            ha="center", va="center",
            fontsize=model_label_fontsize, fontweight='bold',
            transform=ax.transAxes)
    fig2.add_subplot(ax)

last_row_axes = None  # we'll keep the last row's axes to size the colorbar

# Plot each SCD_ms subset row with A1-A4, B1-B4, C1-C4 labels
for row_idx, ((row_label, row_data), vmin, vmax) in enumerate(zip(all_data_scd, vmins_scd, vmaxs_scd)):
    inner2 = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=outer2[row_idx + 1], wspace=0.02)

    row_letter = chr(65 + row_idx)  # A, B, C
    row_axes = []

    for col_idx in range(4):
        ax = plt.Subplot(fig2, inner2[col_idx])
        ax.set_xticks([])
        ax.set_yticks([])

        if col_idx == 0:
            # Original image + row title INSIDE the axis so it never gets cut off
            ax.imshow(row_data[0], cmap="gray")
            ax.text(0.02, 0.5, row_label, va="center", ha="left",
                    fontsize=label_fontsize, transform=ax.transAxes,
                    rotation=90, fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.25", facecolor='white', alpha=0.7, linewidth=0))
        else:
            ax.imshow(row_data[0], cmap="gray")
            ax.imshow(row_data[col_idx], cmap="jet", alpha=0.5,
                      vmin=global_vmin_scd, vmax=global_vmax_scd)

        # A1, A2, A3, A4 labels
        ax.text(0.05, 0.95, f"{row_letter}{col_idx + 1}",
                ha="left", va="top", fontsize=subplot_label_fontsize,
                fontweight='bold', transform=ax.transAxes, color='white',
                bbox=dict(boxstyle="round,pad=0.30", facecolor='black', alpha=0.7, linewidth=0))

        fig2.add_subplot(ax)
        row_axes.append(ax)

    # keep the last row’s axes for colorbar sizing
    if row_idx == len(all_data_scd) - 1:
        last_row_axes = row_axes

# Margins: add a bit more left to make room, and bottom for the colorbar
fig2.subplots_adjust(left=0.08, right=0.98, top=0.96, bottom=0.12)

# --- COLORBAR exactly under the last row, spanning the image width ---
fig2.canvas.draw()  # ensure positions are up-to-date

left_ax = last_row_axes[0].get_position(fig2)
right_ax = last_row_axes[-1].get_position(fig2)

cb_left = left_ax.x0
cb_right = right_ax.x1
cb_width = cb_right - cb_left
cb_height = 0.035  # tweak if you want it thinner/thicker
cb_bottom = left_ax.y0 - 0.06  # distance below last row; adjust if needed

cax2 = fig2.add_axes([cb_left, cb_bottom, cb_width, cb_height])

sm2 = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=global_vmin_scd, vmax=global_vmax_scd))
sm2.set_array([])
cbar2 = fig2.colorbar(sm2, cax=cax2, orientation='horizontal')  # NOTE: use cax=, not ax=

# Tick placement & labels
tick_offset2 = (global_vmax_scd - global_vmin_scd) * 0.10
adjusted_vmin2 = global_vmin_scd + tick_offset2
cbar2.set_ticks([adjusted_vmin2, global_vcenter_scd, global_vmax_scd])
cbar2.ax.set_xticklabels([f"{global_vmin_scd:.3f}", f"{global_vcenter_scd:.3f}", f"{global_vmax_scd:.3f}"],
                         fontsize=colorbar_number_fontsize)
cbar2.ax.tick_params(axis='x', which='major', pad=15, labelsize=colorbar_number_fontsize)
cbar2.set_label("Δ Score", fontsize=label_fontsize)

# Remove colorbar border but keep ticks
for spine in cbar2.ax.spines.values():
    spine.set_visible(False)

# Save
output_path2 = "results/visualize/Heatmaps/combined_occlusion_heatmap_grid_SCDms_rows_with_labels_and_global_colorbar.png"
os.makedirs(os.path.dirname(output_path2), exist_ok=True)
plt.savefig(output_path2, dpi=300)
plt.close()
print(f"[✓] Saved: {output_path2}")