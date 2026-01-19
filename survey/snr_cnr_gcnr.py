#!/usr/bin/env python3
# survey/snr_cnr_gcnr.py
"""
Compute region-based SNR, CNR, and gCNR for a small set of PAI images with robust ROI fallback.

Outputs:
  - survey/snr_cnr_gcnr_results.csv
  - survey/roi_overlays/<basename>_roi.png (visual QA)
  - survey/roi_overlays/<basename>_roi_DEBUG.png when fallbacks were needed

Dependencies:
  pip install numpy pillow scikit-image pandas matplotlib
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
from skimage import filters, morphology, measure
from skimage.exposure import rescale_intensity
from skimage.morphology import disk, remove_small_objects, binary_opening, binary_closing
from scipy.ndimage import distance_transform_edt

# ---- Image list ----
IMAGE_PATHS = [
    "survey/mice_image_113_32.png",
    "survey/mice_image_113_128.png",
    "survey/phantom_image_212_32.png",
    "survey/phantom_image_212_128.png",
    "survey/SWFD_multisegment_ms,ss32_BP_slice0082.png",
    "survey/SWFD_multisegment_ms,ss128_BP_slice0082.png",
    "survey/SWFD_semicircle_sc,ss32_BP_slice0031.png",
    "survey/SWFD_semicircle_sc,ss128_BP_slice0031.png",
]

def load_grayscale(path):
    return np.asarray(Image.open(path).convert("L")).astype(np.float32)

def normalize01(img, clip_percentile=(0.0, 100.0)):
    lo = np.percentile(img, clip_percentile[0])
    hi = np.percentile(img, clip_percentile[1])
    if hi <= lo:
        lo, hi = float(img.min()), float(img.max() + 1e-12)
    out = (img - lo) / (hi - lo + 1e-12)
    return np.clip(out, 0.0, 1.0)

def largest_component(mask, min_size=0):
    if mask.sum() == 0:
        return mask
    labeled = measure.label(mask, connectivity=2)
    props = measure.regionprops(labeled)
    if not props:
        return mask
    largest = max(props, key=lambda p: p.area)
    out = (labeled == largest.label)
    if min_size and out.sum() < min_size:
        return mask  # let caller decide it failed
    return out

def clean_blob(mask):
    mask = binary_opening(mask, footprint=disk(1))
    mask = binary_closing(mask, footprint=disk(1))
    mask = remove_small_objects(mask, min_size=64)
    return mask

def histogram_gcnr(sig_vals, bg_vals, bins=256):
    vmin = min(sig_vals.min(), bg_vals.min())
    vmax = max(sig_vals.max(), bg_vals.max())
    if vmax <= vmin:
        vmax = vmin + 1.0
    hist_s, edges = np.histogram(sig_vals, bins=bins, range=(vmin, vmax), density=True)
    hist_b, _     = np.histogram(bg_vals,  bins=bins, range=(vmin, vmax), density=True)
    overlap = np.minimum(hist_s, hist_b).sum() * (edges[1] - edges[0])
    gcnr = 1.0 - float(overlap)
    return float(np.clip(gcnr, 0.0, 1.0))

def compute_metrics(img_vals, sig_mask, bg_mask, bins=256):
    sig = img_vals[sig_mask].astype(np.float64)
    bg  = img_vals[bg_mask].astype(np.float64)
    snr_linear = (sig.mean() / (bg.std() + 1e-12))
    snr_db = 20.0 * np.log10(max(snr_linear, 1e-12))
    cnr = abs(sig.mean() - bg.mean()) / np.sqrt(sig.var() + bg.var() + 1e-12)
    gcnr = histogram_gcnr(sig, bg, bins=bins)
    return dict(
        SNR_linear=float(snr_linear),
        SNR_dB=float(snr_db),
        CNR=float(cnr),
        gCNR=float(gcnr),
        sig_n=int(sig.size),
        bg_n=int(bg.size),
        sig_mean=float(sig.mean()),
        sig_std=float(sig.std()),
        bg_mean=float(bg.mean()),
        bg_std=float(bg.std()),
    )

def save_overlay(img_norm, sig_mask, bg_mask, out_path, debug=False):
    base = (rescale_intensity(img_norm, in_range=(0, 1), out_range=(0, 255))).astype(np.uint8)
    rgb = np.stack([base, base, base], axis=-1)

    # fill
    rgb[sig_mask, 1] = np.clip(rgb[sig_mask, 1] + 80, 0, 255)  # green
    rgb[bg_mask, 0]  = np.clip(rgb[bg_mask, 0]  + 80, 0, 255)  # red

    # edges
    sig_edge = morphology.binary_dilation(sig_mask) ^ morphology.binary_erosion(sig_mask)
    bg_edge  = morphology.binary_dilation(bg_mask) ^ morphology.binary_erosion(bg_mask)
    rgb[sig_edge] = [0, 255, 0]
    rgb[bg_edge]  = [255, 0, 0]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out_path if not debug else out_path.with_name(out_path.stem + "_DEBUG.png"))

def distance_based_background(valid_mask, sig_mask, min_bg, margin_px=3, img_norm=None):
    """Background = low-intensity pixels sufficiently far from signal."""
    not_sig = ~morphology.binary_dilation(sig_mask, footprint=disk(margin_px))
    candidates = valid_mask & not_sig
    if img_norm is None:
        return candidates
    # Prefer low intensities among candidates
    vals = img_norm[candidates]
    if vals.size < min_bg:
        return candidates
    thr = np.percentile(vals, 50.0)  # bottom half by default
    bg = candidates & (img_norm <= thr)
    if bg.sum() < min_bg:
        thr = np.percentile(vals, 65.0)
        bg = candidates & (img_norm <= thr)
    if bg.sum() < min_bg:
        bg = candidates
    return bg

def progressive_rois(img_norm, min_signal=200, min_bg=200, fallback_k=500):
    """
    Try multiple strategies to guarantee adequate signal/background sizes.
    Returns (sig_mask, bg_mask, note) or (None, None, note) on failure.
    """
    valid = img_norm > 1e-6
    note_parts = []

    # --- A) global thresholds: Otsu, Yen, Triangle
    for name, thr_func in [("Otsu", filters.threshold_otsu),
                           ("Yen", filters.threshold_yen),
                           ("Triangle", filters.threshold_triangle)]:
        try:
            thr = thr_func(img_norm)
        except Exception:
            continue
        raw = img_norm > thr
        sig = clean_blob(raw)
        sig = largest_component(sig)
        if sig.sum() >= min_signal:
            bg = distance_based_background(valid, sig, min_bg, margin_px=3, img_norm=img_norm)
            if bg.sum() >= min_bg:
                note_parts.append(f"{name}")
                return sig, bg, "+".join(note_parts)
            note_parts.append(f"{name}:bg++")

    # --- B) percentile walk for signal & background
    for sig_pct in [99.0, 97.0, 95.0]:
        sig = img_norm >= np.percentile(img_norm[valid], sig_pct)
        sig = clean_blob(sig)
        sig = largest_component(sig)
        if sig.sum() >= min_signal:
            for bg_pct in [20.0, 35.0, 50.0]:
                margin = 3
                not_edge = ~morphology.binary_dilation(sig, footprint=disk(margin))
                bg = (img_norm <= np.percentile(img_norm[valid], bg_pct)) & valid & not_edge
                if bg.sum() < min_bg:
                    bg = distance_based_background(valid, sig, min_bg, margin_px=margin, img_norm=img_norm)
                if bg.sum() >= min_bg:
                    note_parts.append(f"pct{int(sig_pct)}/{int(bg_pct)}")
                    return sig, bg, "+".join(note_parts)
            note_parts.append(f"pct{int(sig_pct)}:bg++")

    # --- C) last resort: pick exactly K brightest for signal and K' darkest for background
    vals = img_norm[valid]
    if vals.size >= (min_signal + min_bg):
        k_sig = max(min_signal, min(fallback_k, vals.size // 4))
        k_bg  = max(min_bg, min(fallback_k * 4, vals.size // 2))
        flat = img_norm.copy().ravel()
        order = np.argsort(flat)  # ascending
        # darkest k_bg
        bg_idx = order[:k_bg]
        # brightest k_sig (exclude any overlap)
        sig_idx = order[-k_sig:]
        sig_mask = np.zeros_like(flat, dtype=bool)
        bg_mask  = np.zeros_like(flat, dtype=bool)
        sig_mask[sig_idx] = True
        bg_mask[bg_idx]   = True
        sig_mask = sig_mask.reshape(img_norm.shape)
        bg_mask  = bg_mask.reshape(img_norm.shape)

        # Make signal blob-like (largest component)
        sig_mask = clean_blob(sig_mask)
        sig_mask = largest_component(sig_mask)
        if sig_mask.sum() < min_signal:
            # if cleaning killed it, revert to raw K-pixels
            sig_mask = np.zeros_like(img_norm, dtype=bool)
            sig_mask.ravel()[sig_idx] = True

        # Ensure background is not touching signal edges
        bg_mask = distance_based_background(valid, sig_mask, min_bg, margin_px=3, img_norm=img_norm)
        if sig_mask.sum() >= min_signal and bg_mask.sum() >= min_bg:
            note_parts.append(f"K({k_sig},{k_bg})")
            return sig_mask, bg_mask, "+".join(note_parts)

    return None, None, "+".join(note_parts) if note_parts else "failed"

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bins", type=int, default=256, help="Histogram bins for gCNR")
    parser.add_argument("--min-signal", type=int, default=200, help="Minimum pixels in signal ROI")
    parser.add_argument("--min-bg", type=int, default=200, help="Minimum pixels in background ROI")
    parser.add_argument("--fallback-k", type=int, default=500, help="K for last-resort pixel picking")
    args = parser.parse_args()

    out_csv = Path("survey/snr_cnr_gcnr_results.csv")
    overlay_dir = Path("survey/roi_overlays")
    overlay_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in IMAGE_PATHS:
        path = Path(p)
        if not path.exists():
            print(f"[WARN] Missing image: {path}")
            continue

        print(f"[INFO] Processing {path} ...")
        img = load_grayscale(path)
        img_norm = normalize01(img)

        sig_mask, bg_mask, note = progressive_rois(
            img_norm, min_signal=args.min_signal, min_bg=args.min_bg, fallback_k=args.fallback_k
        )

        if sig_mask is None or bg_mask is None:
            print(f"  [ERROR] Could not derive ROIs ({note}). Skipping.")
            continue

        metrics = compute_metrics(img_norm, sig_mask, bg_mask, bins=args.bins)
        metrics.update({
            "image_path": str(path),
            "roi_strategy": note,
        })
        rows.append(metrics)

        overlay_path = overlay_dir / f"{path.stem}_roi.png"
        save_overlay(img_norm, sig_mask, bg_mask, overlay_path)
        if "K(" in note or "bg++" in note or "pct" in note or "Yen" in note or "Triangle" in note:
            save_overlay(img_norm, sig_mask, bg_mask, overlay_path, debug=True)

        print(f"  [OK] SNR_dB={metrics['SNR_dB']:.2f}, CNR={metrics['CNR']:.3f}, gCNR={metrics['gCNR']:.3f} "
              f"(sig_n={metrics['sig_n']}, bg_n={metrics['bg_n']}). Strategy={note}")

    if rows:
        df = pd.DataFrame(rows)
        cols = ["image_path", "roi_strategy", "SNR_linear", "SNR_dB", "CNR", "gCNR",
                "sig_n", "bg_n", "sig_mean", "sig_std", "bg_mean", "bg_std"]
        df = df[[c for c in cols if c in df.columns] + [c for c in df.columns if c not in cols]]
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        print(f"\n[RESULT] Saved metrics to: {out_csv}")
        print(f"[RESULT] Saved ROI overlays to: {overlay_dir}")
    else:
        print("\n[RESULT] No rows produced. Check errors above.")

if __name__ == "__main__":
    main()