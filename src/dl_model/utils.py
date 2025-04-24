import os
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
from PIL import Image
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split

class PhotoacousticDataset(Dataset):
    def __init__(self, csv_file, target_metric='CLIP-IQA', image_size=128):
        """
        Args:
            csv_file (str): Path to CSV file with columns: image_path, metric1, metric2, ...
            target_metric (str): The metric to use as a regression target (e.g., 'CLIP-IQA')
            image_size (int): Desired square image size (128x128 default)
        """
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
        image = Image.open(img_path).convert('L')  # grayscale

        # Convert to tensor and resize
        image_tensor = self.to_tensor(image)

        # Normalize to [0, 1] if needed (custom logic to prevent double normalization)
        min_val = image_tensor.min()
        max_val = image_tensor.max()
        if min_val < 0.0 or max_val > 1.0:
            image_tensor = (image_tensor - min_val) / (max_val - min_val + 1e-8)

        target = torch.tensor(self.data.iloc[idx][self.target_metric], dtype=torch.float32)
        return image_tensor, target

def create_train_val_splits(data_dir, val_fraction=0.2, test_fraction=0.1, target_metric="CLIP-IQA"):
    """
    Loads all CSVs in data_dir (recursively), combines them, and splits into train/val/test.
    """
    # Grab all *_per_image_metrics_*.csv recursively
    all_csvs = sorted(glob(os.path.join(data_dir, "**/*_per_image_metrics_*.csv"), recursive=True))
    if not all_csvs:
        raise ValueError(f"No CSVs found in {data_dir}. Expected *_per_image_metrics_*.csv format.")

    # Combine into single DataFrame
    combined_df = pd.concat([pd.read_csv(f) for f in all_csvs], ignore_index=True)

    # First split: train+val vs test
    train_val_df, test_df = train_test_split(combined_df, test_size=test_fraction, random_state=42)

    # Second split: train vs val
    val_size = val_fraction / (1.0 - test_fraction)
    train_df, val_df = train_test_split(train_val_df, test_size=val_size, random_state=42)

    # Wrap into datasets
    train_dataset = PhotoacousticDatasetFromDataFrame(train_df, target_metric=target_metric)
    val_dataset = PhotoacousticDatasetFromDataFrame(val_df, target_metric=target_metric)
    test_dataset = PhotoacousticDatasetFromDataFrame(test_df, target_metric=target_metric)  

    return train_dataset, val_dataset, test_dataset

class PhotoacousticDatasetFromDataFrame(Dataset):
    """
    Same as PhotoacousticDataset, but initialized from an already-loaded DataFrame.
    """
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