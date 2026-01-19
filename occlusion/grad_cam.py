# occlusion/grad_cam.py
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torchvision import transforms

from dl_model.inference import load_model_checkpoint

# ---------------------- Settings ----------------------
device = "cuda"
model_paths = {
    "best_model": "models/best_model_thesis/SSIM/best_model.pth",         # PAQNet
    "IQDCNN": "models/IQDCNN_thesis/SSIM/best_model.pth",
    "EfficientNetIQA": "models/EfficientNetIQA_thesis/SSIM/best_model.pth",
}
model_display = {
    "original": "Original",
    "best_model": "PAQNet",
    "IQDCNN": "IQDCNN",
    "EfficientNetIQA": "EfficientNetIQA",
}

# Fonts
tick_fontsize = 15
label_fontsize = 24
subplot_label_fontsize = 30
model_label_fontsize = 36
colorbar_number_fontsize = 20

# Transform (same as heatmaps)
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
    # Pearson correlation (safe for constant arrays)
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

# ---------------------- Grad-CAM helper ----------------------
class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def remove_hooks(self):
        try:
            self._fwd_hook.remove()
            self._bwd_hook.remove()
        except Exception:
            pass

    def __call__(self, input_tensor: torch.Tensor):
        self.model.zero_grad(set_to_none=True)
        out = self.model(input_tensor)  # [1,1] or [1,K]
        score = out.mean() if out.ndim == 2 else out.squeeze()
        score.backward(retain_graph=False)

        A = self.activations
        dYdA = self.gradients
        if A is None or dYdA is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = dYdA.mean(dim=(2, 3), keepdim=True)
        cam = (weights * A).sum(dim=1, keepdim=False)
        cam = F.relu(cam).squeeze(0)

        H, W = input_tensor.shape[-2:]
        cam = cam.unsqueeze(0).unsqueeze(0)
        cam = F.interpolate(cam, size=(H, W), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy().astype(np.float32)
        # Normalize [0,1] for visualization
        cam = _normalize01(cam)
        return cam

def find_last_conv2d(module: torch.nn.Module) -> torch.nn.Conv2d:
    last_conv = None
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM target.")
    return last_conv

def get_target_layer_for_model(model: torch.nn.Module) -> torch.nn.Module:
    search_root = getattr(model, "model", model)
    return find_last_conv2d(search_root)

# ---------------------- Figure helpers ----------------------
def prepare_row_data_gradcam(img_path, models_by_key, dataset_key, row_label):
    img = Image.open(img_path).convert("L").resize((128, 128))
    original_np = np.array(img, dtype=np.float32) / 255.0
    input_tensor = transform(img).unsqueeze(0).to(device)

    row_data = [original_np]
    cams = []
    for model_key in model_paths:
        model = models_by_key[model_key]
        target_layer = get_target_layer_for_model(model)
        cam_engine = GradCAM(model, target_layer)
        cam = cam_engine(input_tensor)
        cam_engine.remove_hooks()
        cams.append(cam)
        # --- Save raw Grad-CAM map for metrics cross-compare ---
        out_npy = Path(f"occlusion/maps/gradcam/{dataset_key}")
        out_npy.mkdir(parents=True, exist_ok=True)
        np.save(out_npy / f"{row_label}_{model_key}.npy", cam)

        # If corresponding heatmap exists, compute metrics now
        heat_npy = Path(f"occlusion/maps/heatmap/{dataset_key}/{row_label}_{model_key}.npy")
        if heat_npy.exists():
            heat_map = np.load(heat_npy)
            l2, corr = _compare_maps(heat_map, cam)
            _append_metrics_row(dataset_key, row_label, model_key, l2, corr)

    row_data.extend(cams)
    return row_data

def plot_dataset_rows_gradcam(dataset_key, rows, models_by_key):
    all_rows = []
    for row_label, img_path in rows:
        row_data = prepare_row_data_gradcam(img_path, models_by_key, dataset_key, row_label)
        all_rows.append((row_label, row_data))

    fig = plt.figure(figsize=(18, 18))
    outer = gridspec.GridSpec(len(rows) + 1, 1, height_ratios=[0.12] + [1]*len(rows), hspace=0.25)

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
                ax.imshow(row_data[c], cmap="jet", alpha=0.5, vmin=0.0, vmax=1.0)

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
    cb_left = left_ax.x0
    cb_right = right_ax.x1
    cb_width = cb_right - cb_left
    cb_height = 0.035
    cb_bottom = left_ax.y0 - 0.06

    cax = fig.add_axes([cb_left, cb_bottom, cb_width, cb_height])
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation='horizontal')
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.ax.set_xticklabels(["0.000", "0.500", "1.000"], fontsize=colorbar_number_fontsize)
    cbar.ax.tick_params(axis='x', which='major', pad=15, labelsize=colorbar_number_fontsize)
    cbar.set_label("Grad-CAM intensity", fontsize=label_fontsize)
    for spine in cbar.ax.spines.values():
        spine.set_visible(False)

    out_dir = Path("occlusion/gradcam")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"gradcam_{dataset_key}.png"
    fig.savefig(out_path.as_posix(), dpi=300)
    plt.close(fig)
    print(f"[✓] Saved: {out_path}")

# ---------------------- Load models once ----------------------
models = {k: load_model_checkpoint(p, device=device) for k, p in model_paths.items()}

# ---------------------- Grad-CAM only similarity (no heatmaps) ----------------------
from itertools import combinations

def _normalize01(x: np.ndarray) -> np.ndarray:
    mn, mx = float(x.min()), float(x.max())
    if mx - mn < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn + 1e-8)

def _l2_rms(a: np.ndarray, b: np.ndarray) -> float:
    af, bf = a.ravel(), b.ravel()
    return float(np.sqrt(np.mean((af - bf) ** 2)))

def compare_gradcams_inter_model(dataset_key: str, rows: list, model_keys=None):
    """
    For each row (subset) in a dataset, compute pairwise L2 between Grad-CAMs
    from different models on the SAME image.
    Outputs: occlusion/metrics/gradcam_only_similarity_models.csv
    """
    if model_keys is None:
        model_keys = list(model_paths.keys())  # ["best_model", "IQDCNN", "EfficientNetIQA"]

    out_dir = Path("occlusion/metrics"); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "gradcam_only_similarity_models.csv"
    header_needed = not csv_path.exists()

    with open(csv_path, "a") as f:
        if header_needed:
            f.write("dataset,row_label,pair,l2_distance\n")

        for row_label, _ in rows:
            cams = {}
            for mk in model_keys:
                npy_path = Path(f"occlusion/maps/gradcam/{dataset_key}/{row_label}_{mk}.npy")
                if not npy_path.exists():
                    # silently skip if not generated yet
                    continue
                cam = np.load(npy_path)
                cam = _normalize01(cam)
                cams[mk] = cam

            # pairwise over available models
            for m1, m2 in combinations(cams.keys(), 2):
                l2 = _l2_rms(cams[m1], cams[m2])
                pair_name = f"{m1}__vs__{m2}"
                f.write(f"{dataset_key},{row_label},{pair_name},{l2:.6f}\n")

def compare_gradcams_within_model(dataset_key: str, rows: list, model_keys=None):
    """
    For each MODEL, compare Grad-CAMs across different rows (subsets) in the same dataset.
    E.g., for PAQNet: L2( sparse32 vs sparse64 ), L2( sparse32 vs sparse128 ), L2( sparse64 vs sparse128 ).
    Outputs: occlusion/metrics/gradcam_only_similarity_withinmodel.csv
    """
    if model_keys is None:
        model_keys = list(model_paths.keys())

    out_dir = Path("occlusion/metrics"); out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "gradcam_only_similarity_withinmodel.csv"
    header_needed = not csv_path.exists()

    # load all available cams per model
    # cams_by_model: { model_key: { row_label: cam } }
    cams_by_model = {mk: {} for mk in model_keys}
    for row_label, _ in rows:
        for mk in model_keys:
            npy_path = Path(f"occlusion/maps/gradcam/{dataset_key}/{row_label}_{mk}.npy")
            if npy_path.exists():
                cam = np.load(npy_path)
                cams_by_model[mk][row_label] = _normalize01(cam)

    with open(csv_path, "a") as f:
        if header_needed:
            f.write("dataset,model,row_a,row_b,l2_distance\n")

        for mk in model_keys:
            labels = sorted(cams_by_model[mk].keys())
            for ra, rb in combinations(labels, 2):
                l2 = _l2_rms(cams_by_model[mk][ra], cams_by_model[mk][rb])
                f.write(f"{dataset_key},{mk},{ra},{rb},{l2:.6f}\n")

# ---------------------- Run for all datasets ----------------------
if __name__ == "__main__":
    for key, rows in DATASETS.items():
        for _, p in rows:
            if not Path(p).exists():
                raise FileNotFoundError(f"Missing image for {key}: {p}")
        # Make the Grad-CAM panel and save raw CAM npys
        plot_dataset_rows_gradcam(key, rows, models)
        # Grad-CAM only similarities (no heatmaps involved)
        compare_gradcams_inter_model(key, rows)
        compare_gradcams_within_model(key, rows)