import os
import argparse
from glob import glob
from tqdm import tqdm
from PIL import Image
import torch
from dl_model.inference import load_model_checkpoint
from interpretability.heatmap_analysis import generate_occlusion_heatmap, plot_heatmap_on_image


def batch_heatmap_analysis(model_path, image_dir, output_dir, device='cuda', max_images=20):
    os.makedirs(output_dir, exist_ok=True)

    model = load_model_checkpoint(model_path, device=device)
    image_paths = sorted(glob(os.path.join(image_dir, "*.png")))[:max_images]

    print(f"Generating heatmaps for {len(image_paths)} images...")

    for img_path in tqdm(image_paths):
        fname = os.path.basename(img_path).replace(".png", "_heatmap.png")
        save_path = os.path.join(output_dir, fname)

        heatmap, base_score = generate_occlusion_heatmap(model, img_path, device=device)
        plot_heatmap_on_image(img_path, heatmap, save_path)

    return sorted(glob(os.path.join(output_dir, "*_heatmap.png")))


def create_combined_grid(image_paths, output_path, grid_size=(5, 5), img_size=(128, 128)):
    assert len(image_paths) >= grid_size[0] * grid_size[1], "Not enough heatmaps for full grid."

    images = [Image.open(p).resize(img_size).convert("RGB") for p in image_paths[:grid_size[0] * grid_size[1]]]

    grid_img = Image.new("RGB", (img_size[0] * grid_size[1], img_size[1] * grid_size[0]))
    for idx, img in enumerate(images):
        row = idx // grid_size[1]
        col = idx % grid_size[1]
        grid_img.paste(img, (col * img_size[0], row * img_size[1]))

    grid_img.save(output_path)
    print(f"✔ Combined heatmap grid saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--image_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default="results/heatmaps/batch")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--max_images', type=int, default=20)
    args = parser.parse_args()

    heatmap_paths = batch_heatmap_analysis(
        model_path=args.model_path,
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        device=args.device,
        max_images=args.max_images
    )

    if heatmap_paths:
        grid_path = os.path.join(args.output_dir, "heatmap_grid.png")
        create_combined_grid(heatmap_paths, output_path=grid_path)