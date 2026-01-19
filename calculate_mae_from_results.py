"""
Calculate MAE, std, and 95% CI from existing evaluation results.

Reads prediction vs target CSV files and calculates:
- MAE (Mean Absolute Error) = mean(|pred_i - true_i|)
- Standard deviation of absolute errors
- 95% Confidence Interval = 1.96 * std / sqrt(n)
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
MODELS = ["best_model_thesis", "IQDCNN_thesis", "EfficientNetIQA_thesis"]
METRICS = ["S3IM", "SSIM", "IWSSIM", "GMSD_norm", "HAARPSI"]
RESULTS_DIR = "results/eval_model"

# Dataset grouping: datasets that should be combined into one
DATASET_GROUPS = {
    'SCD_ms': ['SCD_ms_ss32', 'SCD_ms_ss64', 'SCD_ms_ss128']
}

def calculate_mae_stats(targets, predictions):
    """
    Calculate MAE, std, and 95% CI from targets and predictions.
    
    Args:
        targets: array of target values
        predictions: array of prediction values
    
    Returns:
        dict with mae, std, ci_95, n_samples
    """
    # Calculate absolute errors
    absolute_errors = np.abs(predictions - targets)
    
    # Remove any NaN or inf values
    valid_mask = np.isfinite(absolute_errors)
    absolute_errors = absolute_errors[valid_mask]
    
    if len(absolute_errors) == 0:
        return None
    
    n_samples = len(absolute_errors)
    mae = np.mean(absolute_errors)
    std = np.std(absolute_errors)
    ci_95 = 1.96 * (std / np.sqrt(n_samples))
    
    return {
        'mae': mae,
        'std': std,
        'ci_95': ci_95,
        'n_samples': n_samples
    }

def extract_targets_predictions(df, metric):
    """
    Extract targets and predictions from a DataFrame.
    
    Args:
        df: pandas DataFrame
        metric: name of the metric
    
    Returns:
        tuple of (targets, predictions) arrays or None if columns not found
    """
    # Handle different CSV formats
    # Check if we have target and prediction columns
    if 'target' in df.columns and 'prediction' in df.columns:
        target_col = 'target'
        pred_col = 'prediction'
    elif 'target_' + metric in df.columns and 'prediction_' + metric in df.columns:
        target_col = 'target_' + metric
        pred_col = 'prediction_' + metric
    else:
        # Try to find columns that match the pattern
        target_cols = [col for col in df.columns if col.startswith('target')]
        pred_cols = [col for col in df.columns if col.startswith('prediction')]
        
        if len(target_cols) == 0 or len(pred_cols) == 0:
            return None, None
        
        # For multi-metric files, use the first matching column
        target_col = target_cols[0]
        pred_col = pred_cols[0]
    
    # Extract targets and predictions
    targets = df[target_col].values.astype(float)
    predictions = df[pred_col].values.astype(float)
    
    return targets, predictions

def process_csv_file(csv_path, model_name, metric, dataset_name):
    """
    Process a single CSV file and return targets and predictions.
    
    Args:
        csv_path: path to the CSV file
        model_name: name of the model
        metric: name of the metric
        dataset_name: name of the dataset
    
    Returns:
        tuple of (targets, predictions) arrays or None if file doesn't exist
    """
    if not os.path.exists(csv_path):
        return None, None
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Extract targets and predictions
        targets, predictions = extract_targets_predictions(df, metric)
        
        if targets is None:
            print(f"  Warning: Could not find target/prediction columns in {csv_path}")
            return None, None
        
        return targets, predictions
    
    except Exception as e:
        print(f"  Error processing {csv_path}: {e}")
        return None, None

def find_datasets(base_path):
    """
    Find all dataset directories for a given model/metric combination.
    
    Args:
        base_path: base path like results/eval_model/{model}/{metric}
    
    Returns:
        list of dataset names
    """
    if not os.path.exists(base_path):
        return []
    
    datasets = []
    for item in os.listdir(base_path):
        item_path = os.path.join(base_path, item)
        if os.path.isdir(item_path):
            # Check if it contains a preds_vs_targets CSV file
            csv_files = [f for f in os.listdir(item_path) if f.startswith('preds_vs_targets') and f.endswith('.csv')]
            if csv_files:
                datasets.append(item)
    return datasets

def combine_datasets(targets_list, predictions_list):
    """
    Combine targets and predictions from multiple datasets.
    
    Args:
        targets_list: list of target arrays
        predictions_list: list of prediction arrays
    
    Returns:
        tuple of (combined_targets, combined_predictions)
    """
    if not targets_list or not predictions_list:
        return None, None
    
    # Filter out None values
    valid_targets = [t for t in targets_list if t is not None]
    valid_predictions = [p for p in predictions_list if p is not None]
    
    if len(valid_targets) == 0 or len(valid_predictions) == 0:
        return None, None
    
    # Combine all arrays
    combined_targets = np.concatenate(valid_targets)
    combined_predictions = np.concatenate(valid_predictions)
    
    return combined_targets, combined_predictions

def main():
    """Main function to process all models, metrics, and datasets."""
    all_results = []
    
    print("Calculating MAE, std, and 95% CI from existing evaluation results...")
    print("=" * 80)
    
    # Create reverse mapping: which datasets belong to which group
    dataset_to_group = {}
    for group_name, datasets in DATASET_GROUPS.items():
        for dataset in datasets:
            dataset_to_group[dataset] = group_name
    
    for model_name in MODELS:
        print(f"\nProcessing model: {model_name}")
        print("-" * 80)
        
        for metric in METRICS:
            print(f"  Metric: {metric}")
            
            # Construct base path
            base_path = os.path.join(RESULTS_DIR, model_name, metric)
            
            if not os.path.exists(base_path):
                print(f"    Warning: Path does not exist: {base_path}")
                continue
            
            # Find all datasets for this model/metric combination
            datasets = find_datasets(base_path)
            
            if not datasets:
                print(f"    No datasets found in {base_path}")
                continue
            
            # Process grouped datasets separately
            grouped_datasets_processed = set()
            
            # First, collect data for grouped datasets
            for group_name, group_datasets in DATASET_GROUPS.items():
                targets_list = []
                predictions_list = []
                found_datasets = []
                
                for dataset_name in group_datasets:
                    if dataset_name not in datasets:
                        continue
                    
                    # Construct CSV file path
                    csv_filename = f"preds_vs_targets_{metric}_{dataset_name}.csv"
                    csv_path = os.path.join(base_path, dataset_name, csv_filename)
                    
                    # Process the CSV file
                    targets, predictions = process_csv_file(csv_path, model_name, metric, dataset_name)
                    
                    if targets is not None and predictions is not None:
                        targets_list.append(targets)
                        predictions_list.append(predictions)
                        found_datasets.append(dataset_name)
                        grouped_datasets_processed.add(dataset_name)
                
                # Combine the datasets and calculate statistics
                if targets_list and predictions_list:
                    combined_targets, combined_predictions = combine_datasets(targets_list, predictions_list)
                    
                    if combined_targets is not None:
                        stats = calculate_mae_stats(combined_targets, combined_predictions)
                        
                        if stats:
                            result = {
                                'model': model_name,
                                'metric': metric,
                                'dataset': group_name,
                                **stats
                            }
                            all_results.append(result)
                            print(f"    {group_name} (combined from {', '.join(found_datasets)}): "
                                  f"MAE = {result['mae']:.4f} ± {result['std']:.4f} "
                                  f"(95% CI: {result['mae']:.4f} ± {result['ci_95']:.4f}, n={result['n_samples']})")
            
            # Process individual (non-grouped) datasets
            for dataset_name in datasets:
                # Skip if already processed as part of a group
                if dataset_name in grouped_datasets_processed:
                    continue
                
                # Construct CSV file path
                csv_filename = f"preds_vs_targets_{metric}_{dataset_name}.csv"
                csv_path = os.path.join(base_path, dataset_name, csv_filename)
                
                # Process the CSV file
                targets, predictions = process_csv_file(csv_path, model_name, metric, dataset_name)
                
                if targets is not None and predictions is not None:
                    stats = calculate_mae_stats(targets, predictions)
                    
                    if stats:
                        result = {
                            'model': model_name,
                            'metric': metric,
                            'dataset': dataset_name,
                            **stats
                        }
                        all_results.append(result)
                        print(f"    {dataset_name}: MAE = {result['mae']:.4f} ± {result['std']:.4f} "
                              f"(95% CI: {result['mae']:.4f} ± {result['ci_95']:.4f}, n={result['n_samples']})")
    
    # Create summary DataFrame
    if not all_results:
        print("\nNo results found!")
        return
    
    df_results = pd.DataFrame(all_results)
    
    # Print summary by model and metric
    print("\n" + "=" * 80)
    print("SUMMARY BY MODEL AND METRIC")
    print("=" * 80)
    
    for model_name in MODELS:
        print(f"\n{model_name}:")
        model_results = df_results[df_results['model'] == model_name]
        
        for metric in METRICS:
            metric_results = model_results[model_results['metric'] == metric]
            if len(metric_results) > 0:
                print(f"  {metric}:")
                for _, row in metric_results.iterrows():
                    print(f"    {row['dataset']:20s}: MAE = {row['mae']:.4f} ± {row['std']:.4f} "
                          f"(95% CI: {row['mae']:.4f} ± {row['ci_95']:.4f}, n={int(row['n_samples'])})")
    
    # Save results to CSV
    output_file = "mae_statistics_summary.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\nResults saved to: {output_file}")
    
    # Create detailed summary with averages across datasets
    print("\n" + "=" * 80)
    print("AVERAGE MAE ACROSS ALL DATASETS (by Model and Metric)")
    print("=" * 80)
    
    summary_data = []
    for model_name in MODELS:
        for metric in METRICS:
            model_metric_results = df_results[
                (df_results['model'] == model_name) & 
                (df_results['metric'] == metric)
            ]
            
            if len(model_metric_results) > 0:
                avg_mae = model_metric_results['mae'].mean()
                avg_std = model_metric_results['std'].mean()
                # For CI, we need to recalculate based on average std and number of datasets
                n_datasets = len(model_metric_results)
                avg_ci_95 = 1.96 * (avg_std / np.sqrt(n_datasets))
                total_samples = model_metric_results['n_samples'].sum()
                
                summary_data.append({
                    'model': model_name,
                    'metric': metric,
                    'n_datasets': n_datasets,
                    'avg_mae': avg_mae,
                    'avg_std': avg_std,
                    'avg_ci_95': avg_ci_95,
                    'total_samples': total_samples
                })
                
                print(f"{model_name:25s} {metric:15s}: "
                      f"MAE = {avg_mae:.4f} ± {avg_std:.4f} "
                      f"(95% CI: {avg_mae:.4f} ± {avg_ci_95:.4f}, "
                      f"n_datasets={n_datasets}, total_samples={int(total_samples)})")
    
    # Save summary
    if summary_data:
        df_summary = pd.DataFrame(summary_data)
        summary_file = "mae_statistics_summary_averaged.csv"
        df_summary.to_csv(summary_file, index=False)
        print(f"\nAveraged summary saved to: {summary_file}")

if __name__ == "__main__":
    main()

