import torch
import piq

def compute_piq_metrics(y_pred, y_true, batch_size=256):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # print("Data range of y_pred before normalization:", y_pred.min().item(), "to", y_pred.max().item())
    # print("Data range of y_true before normalization:", y_true.min().item(), "to", y_true.max().item())

    def normalize_for_piq(tensor):
        tensor = tensor.float()  # Always convert to float32

        min_val = tensor.min()
        max_val = tensor.max()

        if min_val < 0.0 or max_val > 1.0:
            print(f"Normalizing tensor with min={min_val.item():.3f}, max={max_val.item():.3f} to [0, 1]")
            return (tensor - min_val) / (max_val - min_val + 1e-8)

        print("Tensor already in [0, 1] range — no normalization needed.")
        return tensor

    # Convert numpy to tensor and move to GPU
    y_pred = torch.from_numpy(y_pred).unsqueeze(1).to(device)
    y_true = torch.from_numpy(y_true).unsqueeze(1).to(device)

    y_pred = normalize_for_piq(y_pred)
    y_true = normalize_for_piq(y_true)

    # print("y_pred dimensions:", y_pred.shape)
    # print("y_true dimensions:", y_true.shape)
    # print("Data range of one y_pred image:", y_pred[0].min().item(), "to", y_pred[0].max().item())
    # print("Data range of one y_true image:", y_true[0].min().item(), "to", y_true[0].max().item())

    metric_names = [
        "PSNR", "SSIM", "MSSSIM", "IWSSIM", "VIF",
        "GMSD", "MSGMSD", "HAARPSI"
    ]
    all_metrics = {name: [] for name in metric_names}

    for i in range(0, y_pred.shape[0], batch_size):
        pred_batch = y_pred[i:i+batch_size]
        true_batch = y_true[i:i+batch_size]

        try:
            all_metrics["PSNR"].append(piq.psnr(pred_batch, true_batch, data_range=1.0, reduction='none'))
            all_metrics["SSIM"].append(piq.ssim(pred_batch, true_batch, data_range=1.0, reduction='none'))
            all_metrics["MSSSIM"].append(piq.multi_scale_ssim(pred_batch, true_batch, data_range=1.0, reduction='none'))
            all_metrics["IWSSIM"].append(piq.information_weighted_ssim(pred_batch, true_batch, data_range=1.0, reduction='none'))
            all_metrics["VIF"].append(piq.vif_p(pred_batch, true_batch, data_range=1.0, reduction='none'))
            all_metrics["GMSD"].append(piq.gmsd(pred_batch, true_batch, data_range=1.0, reduction='none'))
            all_metrics["MSGMSD"].append(piq.multi_scale_gmsd(pred_batch, true_batch, data_range=1.0, reduction='none'))
            all_metrics["HAARPSI"].append(piq.haarpsi(pred_batch, true_batch, data_range=1.0, reduction='none'))
        except torch.cuda.OutOfMemoryError:
            print(f"[OOM] Skipping batch {i // batch_size} due to memory error.")
            for key in all_metrics:
                all_metrics[key].append(torch.full((pred_batch.shape[0],), float('nan'), device=device))

        torch.cuda.empty_cache()

    # Aggregate and compute mean/std
    mean_metrics = {}
    std_metrics = {}

    for name in metric_names:
        values = torch.cat(all_metrics[name])
        mean_metrics[name] = values.mean().item()
        std_metrics[name] = values.std().item()

    return mean_metrics, std_metrics