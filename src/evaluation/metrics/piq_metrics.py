import torch
import piq

def compute_piq_metrics(y_pred, y_true):
    y_pred = torch.from_numpy(y_pred).float().unsqueeze(1)
    y_true = torch.from_numpy(y_true).float().unsqueeze(1)
    y_pred /= 255.0
    y_true /= 255.0

    metrics = {}
    metrics["PSNR"] = piq.psnr(y_pred, y_true, data_range=1.0).item()
    metrics["SSIM"] = piq.ssim(y_pred, y_true, data_range=1.0)[0].item()
    metrics["MSSSIM"] = piq.multi_scale_ssim(y_pred, y_true, data_range=1.0).item()
    metrics["IWSSIM"] = piq.information_weighted_ssim(y_pred, y_true, data_range=1.0).item()
    metrics["VIF"] = piq.vif_p(y_pred, y_true, data_range=1.0).item()
    metrics["FSIM"] = piq.fsim(y_pred, y_true, data_range=1.0).item()
    metrics["GMSD"] = piq.gmsd(y_pred, y_true, data_range=1.0).item()
    metrics["MSGMSD"] = piq.multi_scale_gmsd(y_pred, y_true, data_range=1.0).item()
    metrics["HAARPSI"] = piq.haarpsi(y_pred, y_true, data_range=1.0).item()
    return metrics