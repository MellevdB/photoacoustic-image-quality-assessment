import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_results(file_path):
    df = pd.read_csv(file_path, delim_whitespace=True)
    
    # Convert numeric columns and handle 'inf' and 'nan'
    for col in df.columns[4:]:  # Skip first 4 non-numeric columns
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows where all values are 1 or inf in relevant columns
    numeric_cols = df.columns[4:]
    df = df[~df[numeric_cols].apply(lambda row: all((row == 1) | np.isinf(row)), axis=1)]
    
    return df

def plot_metrics_per_dataset(df):
    metrics_groups = {
        'FSIM, SSIM, S3IM': ['FSIM_mean', 'SSIM_mean', 'S3IM_mean'],
        'NQM, VIF': ['NQM_mean', 'VIF_mean'],
        'PSNR': ['PSNR_mean'],
        'BRISQUE': ['BRISQUE_mean'],
    }
    
    for dataset in df['Dataset'].unique():
        subset = df[df['Dataset'] == dataset]
        
        for group_name, metrics in metrics_groups.items():
            available_metrics = [m for m in metrics if m in subset.columns]
            if not available_metrics:
                continue
            
            plt.figure(figsize=(12, 6))
            
            x_labels = subset['Configuration']
            x = np.arange(len(x_labels))
            
            for metric in available_metrics:
                std_col = metric.replace('_mean', '_std')
                plt.errorbar(x, subset[metric], yerr=subset[std_col], label=f"{metric}", capsize=3)
            
            plt.xticks(x, x_labels, rotation=90)
            plt.xlabel("Configuration")
            plt.ylabel("Metric Value")
            plt.title(f"{dataset} - {group_name}")
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            
            plt.show()

if __name__ == "__main__":
    file_path = "results/all_datasets/all_datasets_results_2025-02-11_02-16-41.txt"  # Change to your actual file path
    df = read_results(file_path)
    plot_metrics_per_dataset(df)