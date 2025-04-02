import torch
import numpy as np
from evaluation.metrics.fr import compute_fr_metrics
from evaluation.metrics.nr import compute_nr_metrics
from evaluation.metrics.piq_metrics import compute_piq_metrics

def calculate_metrics(y_pred, y_true, metric_type="all"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metrics_mean, metrics_std = {}, {}
    print("Using", device)

    if device.type == "cuda":
        piq_mean, piq_std = compute_piq_metrics(y_pred, y_true)
        metrics_mean.update({k: round(v, 4) for k, v in piq_mean.items()})
        metrics_std.update({k: round(v, 4) for k, v in piq_std.items()})
    else:
        if metric_type in ["fr", "all"]:
            fr = compute_fr_metrics(y_pred, y_true)
            metrics_mean.update({k: round(np.nanmean(v), 4) for k, v in fr.items()})
            metrics_std.update({k: round(np.nanstd(v), 4) for k, v in fr.items()})
        if metric_type in ["nr", "all"]:
            nr = compute_nr_metrics(y_pred)
            metrics_mean.update({k: round(np.nanmean(v), 4) for k, v in nr.items()})
            metrics_std.update({k: round(np.nanstd(v), 4) for k, v in nr.items()})

    return metrics_mean, metrics_std