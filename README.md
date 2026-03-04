# Photoacoustic Image Quality Assessment

This project provides tools for assessing image quality in photoacoustic imaging. It supports both **traditional image quality metrics** (full-reference and no-reference) and **deep learning models** that predict quality scores from images. It also includes interpretability tools (occlusion heatmaps, Grad-CAM) for understanding model behavior.

## License

MIT License. See [LICENSE](LICENSE) for details.

---

## Project Structure

```
src/
├── main.py                 # Entry point for traditional IQA evaluation
├── config/
│   ├── data_config.py      # Dataset paths, configs, ground-truth mappings
│   └── dataset_configs.py  # Named configuration lists per dataset
├── dl_model/               # Deep learning pipeline
│   ├── train.py
│   ├── train_all_metrics.py
│   ├── model_definition.py
│   ├── inference.py
│   ├── eval_model.py
│   └── utils.py
├── evaluation/             # Traditional IQA metrics & datasets
│   ├── eval.py
│   ├── metrics/
│   ├── preprocess/
│   └── datasets/
└── interpretability/       # Visualization & model interpretability
    ├── heatmap_analysis.py
    ├── big_heatmap_plot.py
    ├── big_scatter_plot.py
    ├── boxplot/
    └── occlusion/
```

---

## File Descriptions

### Root

| File | Description |
|------|-------------|
| `main.py` | Main entry point for running traditional image quality metrics. Orchestrates evaluation across datasets (SCD, SWFD, MSFD, mice, phantom, denoising, pa_experiment, zenodo, varied_split), writes results to text files and per-image CSVs, and saves images used during evaluation. Supports `--datasets`, `--metric_type` (fr/nr/all), and `--test` mode. |

---

### `config/`

| File | Description |
|------|-------------|
| `data_config.py` | Central dataset definitions: paths, configs, and ground-truth mappings for all photoacoustic datasets (SCD, SWFD, MSFD, mice, phantom, v_phantom, denoising, pa_experiment, zenodo, varied_split). Defines `DATA_DIR` and `RESULTS_DIR`. |
| `dataset_configs.py` | Named configuration lists (denoising SNR levels, mice sparse levels, phantom configs, SCD/SWFD/MSFD configs, etc.) used for dataset-specific processing and evaluation. |

---

### `dl_model/` — Deep Learning

| File | Description |
|------|-------------|
| `train.py` | Training loop for quality prediction models. Supports train/val/test splits, multiple model variants (PAQNet, IQDCNN, EfficientNetIQA), loss functions (L1, MSE, Huber, antibias), early stopping, and learning rate scheduling. Saves checkpoints and `train_val_loss.csv`. |
| `train_all_metrics.py` | Batch training script: loops over metrics (e.g. S3IM, SSIM) and model types (PAQNet, IQDCNN, EfficientNetIQA), calls `train_model`, and saves models under `models/shuffled_70_15_15/<model>/<metric>/`. |
| `model_definition.py` | Neural network definitions: `PhotoacousticQualityNet` (dropout/BN/Multi), `IQDCNN` (single and multi-output), `EfficientNetIQA` (single and multi-output). All accept grayscale images and output quality scores. |
| `inference.py` | `load_model_checkpoint` reconstructs a model from a saved checkpoint; `run_inference` runs a trained model on a DataLoader and returns predictions. |
| `eval_model.py` | Evaluates trained models on test sets. Loads checkpoints, runs inference, produces scatter plots (predicted vs target), computes MAE, L1/MSE loss, Spearman/Pearson correlations, and saves CSVs and loss curves. |
| `utils.py` | `PhotoacousticDatasetFromDataFrame`: PyTorch Dataset loading images from a DataFrame with metric targets. `create_train_val_test_split`: builds curated or shuffled (70/15/15) train/val/test splits from per-image CSV files, with path-fix logic for MSFD/SCD/SWFD. |

---

### `evaluation/` — Traditional IQA

#### `evaluation/` (root)

| File | Description |
|------|-------------|
| `eval.py` | Router that dispatches to the correct dataset processor (zenodo, denoising, pa_experiment, MSFD, SCD/SWFD, mat-based, varied_split) and aggregates metric results. |

#### `evaluation/metrics/`

| File | Description |
|------|-------------|
| `calculate.py` | Central metric interface. `calculate_metrics` invokes FR/NR metrics (CPU) or PIQ metrics (GPU when available), returns mean/std/raw per-image arrays. |
| `fr.py` | Full-reference metrics: PSNR, SSIM, VIF, FSIM, UQI, S3IM. Implements S3IM and helpers. |
| `nr.py` | No-reference metrics: BRISQUE, and optionally NIQE/NIQE-K for quality assessment without a reference. |
| `piq_metrics.py` | GPU-accelerated metrics via the PIQ library: PSNR, SSIM, MSSSIM, IWSSIM, VIF, FSIM, GMSD, MSGMSD, HAARPSI, UQI, S3IM, TV, BRISQUE, CLIP-IQA. Used when CUDA is available. |
| `imquality/__init__.py` | Package metadata (version). |
| `imquality/brisque.py` | BRISQUE implementation: feature extraction, SVM model loading, score computation. |
| `imquality/utils.py` | Image loading and PIL-to-numpy conversion helpers for BRISQUE. |
| `imquality/statistics.py` | Statistical helpers for BRISQUE: AsymmetricGeneralizedGaussian, kernels, distribution fitting. |

#### `evaluation/preprocess/`

| File | Description |
|------|-------------|
| `image_loader.py` | Loads image stacks from a folder; returns arrays and image IDs. |
| `normalize.py` | `min_max_normalize_per_image`: normalizes each image in a batch to [0, 1] per-image. |
| `scale_clip.py` | `scale_and_clip`: scales beamformed data by max and clips. |
| `filterBandPass.py` | Bandpass filtering for signals (Butterworth filter). |

#### `evaluation/datasets/`

| File | Description |
|------|-------------|
| `zenodo.py` | Processes Zenodo dataset: loads reference and algorithm outputs, normalizes, computes metrics, and runs correlation analysis. |
| `denoising.py` | Processes denoising data (NNE subset): loads noisy vs ground-truth images, computes IQA metrics. |
| `pa_experiment.py` | Processes PA experiment data (Training subset): loads PA1–PA7 images per category (KneeSlice1, Phantoms, etc.), applies category-specific cropping, computes metrics. |
| `scd_swfd.py` | Processes SCD and SWFD HDF5 data: loads reconstructed and ground-truth arrays, normalizes, computes metrics per configuration. |
| `msfd.py` | Processes MSFD HDF5 data for multi-wavelength images: loads sparse vs full per wavelength, computes metrics. |
| `mat_based.py` | Processes MATLAB `.mat` datasets (mice, phantom, v_phantom): loads recon and GT arrays, computes metrics. |
| `varied_split.py` | Processes varied-split dataset (scene folders with `.webp` images): uses expert ground-truth CSV to select reference images, compares other images to them, computes PIQ metrics. |

---

### `interpretability/` — Visualization & Interpretability

| File | Description |
|------|-------------|
| `heatmap_analysis.py` | Generates occlusion sensitivity heatmaps: systematically occludes patches, measures prediction drop, and visualizes which regions the model relies on. Can run standalone for specific models and images. |
| `big_heatmap_plot.py` | Produces a multi-row, multi-column figure of occlusion heatmaps across models (PAQNet, IQDCNN, EfficientNetIQA) and datasets (Mice, SWFD, SCD, Phantoms) for publication figures. |
| `big_scatter_plot.py` | Builds scatter-plot grids (target vs predicted) for a metric (e.g. S3IM, SSIM) across models and dataset groups. Reads evaluation CSVs from `results/eval_model/` and saves publication-ready figures. |

#### `interpretability/boxplot/`

| File | Description |
|------|-------------|
| `boxplot.py` | Creates boxplots of per-image IQA metric scores by dataset and configuration. Loads per-image CSVs from `results/`, applies manual config ordering, runs t-tests for significance, and saves combined boxplot PNGs plus `significance_ttest_results.csv`. |

#### `interpretability/occlusion/`

| File | Description |
|------|-------------|
| `grad_cam.py` | Combines Grad-CAM (for EfficientNetIQA) and occlusion heatmaps. Generates both, compares them, and produces side-by-side visualizations across datasets (SCD, SWFD, Mice, Phantom). |
| `heatmap.py` | Generates occlusion sensitivity heatmaps for a set of models and images, optionally compares to Grad-CAM, saves similarity metrics to CSV. Uses predefined dataset image paths. |
