import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from glob import glob
from sklearn.model_selection import train_test_split

class PhotoacousticDataset(Dataset):
    def __init__(self, csv_file, target_metric='CLIP-IQA', image_size=128):
        self.data = pd.read_csv(csv_file)
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
        image = Image.open(img_path).convert('L')  # Grayscale

        image_tensor = self.to_tensor(image)

        # Normalize if needed
        min_val = image_tensor.min()
        max_val = image_tensor.max()
        if min_val < 0.0 or max_val > 1.0:
            image_tensor = (image_tensor - min_val) / (max_val - min_val + 1e-8)

        target = torch.tensor(self.data.iloc[idx][self.target_metric], dtype=torch.float32)
        return image_tensor, target

def create_train_val_test_split(data_dir, val_fraction=0.2, target_metric="CLIP-IQA"):
    """
    Custom function to manually split datasets:
    - Train/Val = denoising_data + SWFD + SCD
    - Test = mice + phantom + v_phantom + MSFD + pa_experiment_data + zenodo
    """

    # Manually find CSVs
    all_csvs = sorted(glob(os.path.join(data_dir, "**/*_per_image_metrics_*.csv"), recursive=True))
    if not all_csvs:
        raise ValueError(f"No CSVs found in {data_dir}. Expected *_per_image_metrics_*.csv format.")

    train_val_datasets = []
    test_datasets = []

    for csv_path in all_csvs:
        dataset_name = csv_path.split(os.sep)[-2]  # Assume /dataset_name/file.csv structure

        if dataset_name in ["denoising_data", "SWFD", "SCD"]:
            train_val_datasets.append(csv_path)
        elif dataset_name in ["mice", "phantom", "v_phantom", "MSFD", "pa_experiment_data", "zenodo"]:
            test_datasets.append(csv_path)
        else:
            print(f"[Warning] Skipping unknown dataset: {dataset_name}")

    # Load and combine
    train_val_df = pd.concat([pd.read_csv(f) for f in train_val_datasets], ignore_index=True)
    test_df = pd.concat([pd.read_csv(f) for f in test_datasets], ignore_index=True)

    # Split train and val
    train_df, val_df = train_test_split(train_val_df, test_size=val_fraction, random_state=42)

    # Build datasets
    train_dataset = PhotoacousticDatasetFromDataFrame(train_df, target_metric=target_metric)
    val_dataset = PhotoacousticDatasetFromDataFrame(val_df, target_metric=target_metric)
    test_dataset = PhotoacousticDatasetFromDataFrame(test_df, target_metric=target_metric)

    print(f"Loaded {len(train_dataset)} train samples, {len(val_dataset)} val samples, {len(test_dataset)} test samples.")

    return train_dataset, val_dataset, test_dataset

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
        image = Image.open(img_path).convert('L')

        image_tensor = self.to_tensor(image)

        min_val = image_tensor.min()
        max_val = image_tensor.max()
        if min_val < 0.0 or max_val > 1.0:
            image_tensor = (image_tensor - min_val) / (max_val - min_val + 1e-8)

        target = torch.tensor(self.data.iloc[idx][self.target_metric], dtype=torch.float32)
        return image_tensor, target