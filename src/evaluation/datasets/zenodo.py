import os
import cv2
import numpy as np
from evaluation.metrics.calculate import calculate_metrics

def process_zenodo_data(dataset_info, results, metric_type):
    print(dataset_info["path"])
    print(dataset_info["reference"])
    reference_path = os.path.join(dataset_info["path"], dataset_info["reference"])
    algorithms_path = os.path.join(dataset_info["path"], dataset_info["algorithms"])

    reference_images = sorted([f for f in os.listdir(reference_path) if f.endswith(".png")])
    y_true_stack = [cv2.imread(os.path.join(reference_path, f), cv2.IMREAD_GRAYSCALE) for f in reference_images]
    y_true_stack = np.stack(y_true_stack, axis=0)
    print("Stacked reference images:", y_true_stack.shape)

    for category in dataset_info["categories"]:
        y_pred_stack = []
        for ref_img in reference_images:
            number = ref_img.replace("image", "").replace(".png", "")
            pred_img = f"image{number}_{category}.png"
            pred_path = os.path.join(algorithms_path, pred_img)
            if not os.path.exists(pred_path):
                continue
            img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
            y_pred_stack.append(img)

        if y_pred_stack:
            y_pred_stack = np.stack(y_pred_stack, axis=0)
            print(f"Stacked predicted images for category {category}:", y_pred_stack.shape)
            metrics_mean, metrics_std, metric_score_per_image = calculate_metrics(y_pred_stack, y_true_stack, metric_type)
            analyze_zenodo_correlation(
                metric_score_per_image,
                reference_images,
                category,
                dataset_info["path"]
            )

            results.append((f"method_{category}", "reference", (metrics_mean, metrics_std)))



import os
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

import os
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_zenodo_correlation(metric_score_per_image, reference_images, category, base_path):
    # Load expert scores
    expert_path = "data/zenodo/Overall_Quality.csv"
    expert_df = pd.read_csv(expert_path, sep=";")
    
    # Compute the expert average for later correlation, but keep individual scores for difference analysis
    expert_df["ExpertAvg"] = expert_df[["overall_quality_1", "overall_quality_2"]].mean(axis=1)
    
    # Create DataFrame from metric scores
    df = pd.DataFrame(metric_score_per_image)
    # Strip leading "image" and ".png" to get the number, then reformat the actual filename with category
    numbers = [f.replace("image", "").replace(".png", "") for f in reference_images]
    df["filename"] = [f"image{n}_{category}.png" for n in numbers]
    df = df.merge(expert_df, on="filename")
    
    # Calculate correlations between each metric and the expert average score
    correlations = {}
    for metric in metric_score_per_image.keys():
        corr, _ = spearmanr(df[metric], df["ExpertAvg"])
        correlations[metric] = corr

    print(f"\n📊 Correlations for category {category}:")
    for k, v in correlations.items():
        print(f"{k}: {v:.3f}")

    # Create a subfolder for this category (for scatter plots and CSV)
    save_dir = os.path.join("results", "zenodo", "correlations")
    category_dir = os.path.join(save_dir, category)
    os.makedirs(category_dir, exist_ok=True)

    # Save correlation table as CSV in the category folder
    corr_df = pd.DataFrame.from_dict(correlations, orient="index", columns=["Spearman"])
    corr_csv_path = os.path.join(category_dir, f"correlation_method_{category}.csv")
    corr_df.to_csv(corr_csv_path)
    print(f"Saved correlation CSV at: {corr_csv_path}")

    # Compute and print expert score range for the expert average (if needed)
    expert_avg_range = df["ExpertAvg"].max() - df["ExpertAvg"].min()
    print(f"Expert average score range for category {category}: {expert_avg_range:.3f}")

    # Calculate expert score differences (absolute difference between the two expert scores)
    df["ExpertDiff"] = abs(df["overall_quality_1"] - df["overall_quality_2"])
    # Compute summary statistics for the expert differences
    mean_diff = df["ExpertDiff"].mean()
    min_diff = df["ExpertDiff"].min()
    max_diff = df["ExpertDiff"].max()

    diff_summary = (
        f"Expert Score Difference Statistics for category {category}:\n"
        f"Mean Difference: {mean_diff:.3f}\n"
        f"Minimum Difference: {min_diff:.3f}\n"
        f"Maximum Difference: {max_diff:.3f}\n"
    )
    print(diff_summary)
    
    # Save the expert difference summary to a text file in the category folder
    diff_txt_path = os.path.join(category_dir, "expert_difference.txt")
    with open(diff_txt_path, "w") as f:
        f.write(diff_summary)
    print(f"Saved expert difference summary at: {diff_txt_path}")

    # For each metric, create a scatter plot of ExpertAvg vs. metric score
    for metric in metric_score_per_image.keys():
        plt.figure(figsize=(8, 6))
        sns.regplot(x="ExpertAvg", y=metric, data=df, scatter_kws={"alpha":0.7, "edgecolor":'w', "s":80})
        plt.xlabel("Average Expert Score")
        plt.ylabel(f"{metric} Score")
        plt.title(f"{metric} vs. Expert Score for {category}")
        plt.grid(True)
        plt.tight_layout()
        plot_path = os.path.join(category_dir, f"{metric}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved scatter plot for {metric} at: {plot_path}")
    
    
    analyze_with_polynomial_features(df, category_dir, degree=2)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import pandas as pd
import os


def save_polynomial_predictions_plot(df, y_true, y_pred, category_dir, degree):
    """
    Save a scatter plot comparing predicted vs actual expert scores
    """
    plt.figure(figsize=(8, 6))
    sns.regplot(x=y_true, y=y_pred, scatter_kws={"alpha":0.7, "s":60, "edgecolor":"w"})
    plt.xlabel("Actual Expert Score")
    plt.ylabel("Predicted Score (Polynomial Model)")
    plt.title(f"Predicted vs Actual Expert Score (Degree {degree})")
    plt.grid(True)
    plt.tight_layout()
    save_path = os.path.join(category_dir, f"predicted_vs_actual_degree{degree}.png")
    plt.savefig(save_path)
    plt.close()
    print(f"📈 Saved predicted vs actual plot at: {save_path}")

def extract_weighted_formula(poly, linreg, metrics):
    """
    Convert polynomial regression model into a readable weighted formula.
    """
    terms = poly.get_feature_names_out(metrics)
    weights = linreg.coef_
    intercept = linreg.intercept_

    formula = "score = "
    formula += " + ".join([f"{w:.3f}*{t}" for w, t in zip(weights, terms)])
    formula += f" + {intercept:.3f}"
    return formula

# Updated wrapper to call the plot + formula
def analyze_with_polynomial_features(df, category_dir, expert_col="ExpertAvg", degree=2):
    metrics = ["PSNR", "SSIM", "MSSSIM", "IWSSIM", "VIF", "GMSD", "HAARPSI"]
    df_clean = df.dropna(subset=metrics + [expert_col]).copy()

    X = df_clean[metrics].values
    y = df_clean[expert_col].values

    # Create polynomial feature pipeline
    poly_pipeline = Pipeline([
        ('poly', PolynomialFeatures(degree=degree, interaction_only=False, include_bias=False)),
        ('linreg', LinearRegression())
    ])

    # Fit the model
    poly_pipeline.fit(X, y)
    y_pred = poly_pipeline.predict(X)
    r2 = r2_score(y, y_pred)

    spearman_corr, _ = spearmanr(y, y_pred)
    print(f"📊 Spearman correlation (degree {degree}): {spearman_corr:.4f}")

    # Extract features and coefficients
    poly = poly_pipeline.named_steps['poly']
    linreg = poly_pipeline.named_steps['linreg']
    feature_names = poly.get_feature_names_out(metrics)

    coeff_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": linreg.coef_
    }).sort_values(by="Coefficient", key=lambda x: abs(x), ascending=False)

    coeff_df["Intercept"] = linreg.intercept_
    coeff_df["R2_Score"] = r2
    coeff_df["Spearman"] = spearman_corr

    # Save CSV
    save_path = os.path.join(category_dir, f"polynomial_regression_degree{degree}.csv")
    coeff_df.to_csv(save_path, index=False)
    print(f"✅ Saved polynomial regression results to: {save_path}")

    # Plot predictions vs ground truth
    save_polynomial_predictions_plot(df_clean, y, y_pred, category_dir, degree)

    # Extract readable formula
    formula = extract_weighted_formula(poly, linreg, metrics)
    formula_path = os.path.join(category_dir, f"composite_metric_formula_degree{degree}.txt")
    with open(formula_path, "w") as f:
        f.write(formula)
    print(f"🧮 Saved composite metric formula at: {formula_path}")

    return coeff_df, r2, spearman_corr, formula