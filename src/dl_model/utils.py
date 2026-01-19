import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split

class PhotoacousticDatasetFromDataFrame(Dataset):
    def __init__(self, dataframe, target_metric='CLIP-IQA', image_size=128):
        self.data = dataframe.reset_index(drop=True)
        self.image_size = image_size
        self.target_metric = target_metric

        self.to_tensor = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]['image_path']
        image = Image.open(img_path).convert('L')  # grayscale
        image_tensor = self.to_tensor(image)

        if isinstance(self.target_metric, list):  # multi-output
            targets = torch.tensor(
                [self.data.iloc[idx][metric] for metric in self.target_metric],
                dtype=torch.float32
            )
        else:
            targets = torch.tensor(self.data.iloc[idx][self.target_metric], dtype=torch.float32)

        return image_tensor, targets, img_path 

def create_train_val_test_split(
    data_dir,
    target_metric="CLIP-IQA",
    split_mode="curated",
    val_size=0.15,
    test_size=0.15,
    random_state=42
):
    """
    Updated split:

    - Train: MSFD + SWFD_ms + phantom
    - Val: denoising_data (NNE) + v_phantom
    - Test: SCD (vc + ms) + SWFD_sc + mice + pa_experiment_data
    """

    def load_dataset(csv_path, fix_msfd=False, fix_scd=False, fix_swfd=False):
        df = pd.read_csv(csv_path)

        if fix_msfd:
            df["image_path"] = df["image_path"].apply(
                lambda x: x.replace("results/MSFD_full_", "/projects/prjs1596/photoacoustic/data/OADAT-full/MSFD_full_")
                          .replace("/images_used", "_images_used")
            )

        if fix_scd:
            df["image_path"] = df["image_path"].apply(
                lambda x: x.replace("results/SCD_full/images_used", "/projects/prjs1596/photoacoustic/data/OADAT-full/SCD_full_images_used")
            )

        if fix_swfd:
            df["image_path"] = df["image_path"].apply(
                lambda x: x.replace("results/SWFD_multisegment_ss_full/images_used", "/projects/prjs1596/photoacoustic/data/OADAT-full/SWFD_multisegment_ss_full_images_used")
            )

        return df

    if split_mode == "curated":
        # === TRAIN SET ===
        train_dfs = []

        msfd_paths = [
            "results/MSFD_full_w700/MSFD_full_w700_per_image_metrics_2025-06-09_16-14-04.csv",
            "results/MSFD_full_w730/MSFD_full_w730_per_image_metrics_2025-06-09_16-10-26.csv",
            "results/MSFD_full_w760/MSFD_full_w760_per_image_metrics_2025-06-09_16-10-26.csv",
            "results/MSFD_full_w780/MSFD_full_w780_per_image_metrics_2025-06-09_16-10-32.csv",
            "results/MSFD_full_w800/MSFD_full_w800_per_image_metrics_2025-06-09_16-11-03.csv",
            "results/MSFD_full_w850/MSFD_full_w850_per_image_metrics_2025-06-09_16-12-03.csv"
        ]
        for csv_path in msfd_paths:
            train_dfs.append(load_dataset(csv_path, fix_msfd=True))

        train_dfs.append(load_dataset(
            "results/SWFD_multisegment_ss_full/SWFD_multisegment_ss_full_per_image_metrics_2025-06-08_13-18-56.csv",
            fix_swfd=True
        ))

        train_dfs.append(load_dataset("results/phantom/phantom_per_image_metrics_2025-05-20_13-29-51.csv"))

        train_df = pd.concat(train_dfs, ignore_index=True)
        train_dataset = PhotoacousticDatasetFromDataFrame(train_df, target_metric=target_metric)

        # === VAL SET ===
        val_dfs = [
            load_dataset("results/denoising_data/denoising_data_per_image_metrics_2025-05-20_13-29-51.csv"),
            load_dataset("results/v_phantom/v_phantom_per_image_metrics_2025-05-20_13-29-51.csv")
        ]
        val_df = pd.concat(val_dfs, ignore_index=True)
        val_dataset = PhotoacousticDatasetFromDataFrame(val_df, target_metric=target_metric)

        # === TEST SET ===
        test_datasets = {}

        test_info = {
            "SCD_vc_ms": "results/SCD_full/SCD_full_per_image_metrics_2025-06-08_12-40-15.csv",
            "SCD_ms_ss32": "results/SCD_ms_ss32_full/SCD_ms_ss32_full_per_image_metrics_2025-06-10_07-58-12.csv",
            "SCD_ms_ss64": "results/SCD_ms_ss64_full/SCD_ms_ss64_full_per_image_metrics_2025-06-10_07-57-19.csv",
            "SCD_ms_ss128": "results/SCD_ms_ss128_full/SCD_ms_ss128_full_per_image_metrics_2025-06-09_13-41-48.csv",
            "SWFD_sc": "results/SWFD_semicircle_full/SWFD_semicircle_full_per_image_metrics_2025-06-09_11-12-34.csv",
            "mice": "results/mice/mice_per_image_metrics_2025-05-20_13-29-51.csv",
            "pa_experiment_data": "results/pa_experiment_data/pa_experiment_data_per_image_metrics_2025-05-20_13-29-51.csv",
            "varied_split": "results/varied_split/varied_split_per_image_metrics_expertGT_2025-06-19_11-57-14.csv"

        }

        for name, csv_path in test_info.items():
            df = load_dataset(csv_path, fix_scd=("SCD" in name))
            test_datasets[name] = PhotoacousticDatasetFromDataFrame(df, target_metric=target_metric)

        print(f"Loaded {len(train_dataset)} train, {len(val_dataset)} val, {sum(len(v) for v in test_datasets.values())} test samples from {list(test_datasets.keys())}")

        return train_dataset, val_dataset, test_datasets

    elif split_mode == "shuffled":
        # Collect all available CSVs from the curated configuration, then do a global 70/15/15 shuffle-split
        all_csvs_with_flags = []

        # MSFD (train in curated)
        msfd_paths = [
            "results/MSFD_full_w700/MSFD_full_w700_per_image_metrics_2025-06-09_16-14-04.csv",
            "results/MSFD_full_w730/MSFD_full_w730_per_image_metrics_2025-06-09_16-10-26.csv",
            "results/MSFD_full_w760/MSFD_full_w760_per_image_metrics_2025-06-09_16-10-26.csv",
            "results/MSFD_full_w780/MSFD_full_w780_per_image_metrics_2025-06-09_16-10-32.csv",
            "results/MSFD_full_w800/MSFD_full_w800_per_image_metrics_2025-06-09_16-11-03.csv",
            "results/MSFD_full_w850/MSFD_full_w850_per_image_metrics_2025-06-09_16-12-03.csv"
        ]
        for p in msfd_paths:
            all_csvs_with_flags.append((p, {"fix_msfd": True}))

        # SWFD multisegment (train in curated)
        all_csvs_with_flags.append((
            "results/SWFD_multisegment_ss_full/SWFD_multisegment_ss_full_per_image_metrics_2025-06-08_13-18-56.csv",
            {"fix_swfd": True}
        ))

        # Phantom (train in curated)
        all_csvs_with_flags.append((
            "results/phantom/phantom_per_image_metrics_2025-05-20_13-29-51.csv",
            {}
        ))

        # Val datasets (curated)
        all_csvs_with_flags.append(("results/denoising_data/denoising_data_per_image_metrics_2025-05-20_13-29-51.csv", {}))
        all_csvs_with_flags.append(("results/v_phantom/v_phantom_per_image_metrics_2025-05-20_13-29-51.csv", {}))

        # Test datasets (curated)
        all_csvs_with_flags.append(("results/SCD_full/SCD_full_per_image_metrics_2025-06-08_12-40-15.csv", {"fix_scd": True}))
        all_csvs_with_flags.append(("results/SCD_ms_ss32_full/SCD_ms_ss32_full_per_image_metrics_2025-06-10_07-58-12.csv", {"fix_scd": True}))
        all_csvs_with_flags.append(("results/SCD_ms_ss64_full/SCD_ms_ss64_full_per_image_metrics_2025-06-10_07-57-19.csv", {"fix_scd": True}))
        all_csvs_with_flags.append(("results/SCD_ms_ss128_full/SCD_ms_ss128_full_per_image_metrics_2025-06-09_13-41-48.csv", {"fix_scd": True}))
        all_csvs_with_flags.append(("results/SWFD_semicircle_full/SWFD_semicircle_full_per_image_metrics_2025-06-09_11-12-34.csv", {}))
        all_csvs_with_flags.append(("results/mice/mice_per_image_metrics_2025-05-20_13-29-51.csv", {}))
        all_csvs_with_flags.append(("results/pa_experiment_data/pa_experiment_data_per_image_metrics_2025-05-20_13-29-51.csv", {}))
        all_csvs_with_flags.append(("results/varied_split/varied_split_per_image_metrics_expertGT_2025-06-19_11-57-14.csv", {}))

        # Load and concat
        all_dfs = []
        for path, flags in all_csvs_with_flags:
            df_part = load_dataset(path, fix_msfd=flags.get("fix_msfd", False), fix_scd=flags.get("fix_scd", False), fix_swfd=flags.get("fix_swfd", False))
            all_dfs.append(df_part)
        all_df = pd.concat(all_dfs, ignore_index=True)

        # Drop rows missing the target metric to avoid NaNs during training
        if isinstance(target_metric, list):
            all_df = all_df.dropna(subset=target_metric)
        else:
            all_df = all_df.dropna(subset=[target_metric])

        # First split: train vs temp (val+test)
        train_df, temp_df = train_test_split(
            all_df,
            test_size=val_size + test_size,
            random_state=random_state,
            shuffle=True
        )

        # Second split: val vs test from temp
        relative_test_size = test_size / (val_size + test_size) if (val_size + test_size) > 0 else 0.5
        val_df, test_df = train_test_split(
            temp_df,
            test_size=relative_test_size,
            random_state=random_state,
            shuffle=True
        )

        train_dataset = PhotoacousticDatasetFromDataFrame(train_df, target_metric=target_metric)
        val_dataset = PhotoacousticDatasetFromDataFrame(val_df, target_metric=target_metric)
        test_dataset = PhotoacousticDatasetFromDataFrame(test_df, target_metric=target_metric)

        print(f"[shuffled split] Loaded {len(train_dataset)} train, {len(val_dataset)} val, {len(test_dataset)} test samples")

        # For API compatibility with callers expecting dict of test sets, wrap in a dict
        return train_dataset, val_dataset, {"shuffled_test": test_dataset}

    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")