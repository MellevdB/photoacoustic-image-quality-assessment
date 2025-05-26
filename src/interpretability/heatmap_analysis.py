import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from torchvision import transforms
from PIL import Image
import os
from dl_model.inference import load_model_checkpoint


def generate_occlusion_heatmap(model, image_path, device='cuda', patch_size=16, stride=8):
    # Load image and preprocess
    image = Image.open(image_path).convert('L')
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)  # shape: (1, 1, H, W)

    model.eval()
    with torch.no_grad():
        baseline_pred = model(image_tensor).item()

    _, _, H, W = image_tensor.shape
    heatmap = np.zeros((H, W))
    counts = np.zeros((H, W))

    for i in range(0, H - patch_size + 1, stride):
        for j in range(0, W - patch_size + 1, stride):
            occluded = image_tensor.clone()
            occluded[:, :, i:i+patch_size, j:j+patch_size] = 0.0  # Mask patch with black

            with torch.no_grad():
                pred = model(occluded).item()

            score_drop = baseline_pred - pred
            heatmap[i:i+patch_size, j:j+patch_size] += score_drop
            counts[i:i+patch_size, j:j+patch_size] += 1

    # Avoid division by zero
    counts[counts == 0] = 1
    heatmap /= counts

    return heatmap, baseline_pred


def plot_heatmap_on_image(image_path, heatmap, save_path):
    original = Image.open(image_path).convert('L').resize((128, 128))
    original = np.array(original) / 255.0

    os.makedirs(os.path.dirname(save_path), exist_ok=True)  # ← Add this line

    plt.figure(figsize=(6, 6))
    plt.imshow(original, cmap='gray')
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.colorbar(label='Score Drop')
    plt.title('Occlusion Sensitivity Heatmap')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, default="heatmap.png")
    parser.add_argument('--device', type=str, default="cuda")
    args = parser.parse_args()

    model = load_model_checkpoint(args.model_path, device=args.device)
    heatmap, base_score = generate_occlusion_heatmap(model, args.image_path, device=args.device)
    print(f"Baseline score: {base_score:.4f}")
    plot_heatmap_on_image(args.image_path, heatmap, args.save_path)
    print(f" Heatmap saved to {args.save_path}")
