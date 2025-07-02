import os
import json
import csv
import cv2
import numpy as np


CONFIG_DIR = "src/config"

def map_experiment_data_structure(root_dir):
    """
    Maps the dataset structure in `data/pa_experiment_data/`, ensuring expected categories and files exist.

    :param root_dir: Path to dataset directory.
    :return: Dictionary containing structure mapping.
    """
    dataset_mapping = {}

    # Expected main directories
    expected_main_dirs = ["Training", "Testing"]
    expected_training_subdirs = ["KneeSlice1", "Phantoms", "SmallAnimal", "Transducers"]
    expected_testing_subdirs = ["Invivo", "Phantoms"]

    # Walk through the main directories
    for main_dir in expected_main_dirs:
        main_path = os.path.join(root_dir, main_dir)
        if not os.path.exists(main_path):
            print(f"Missing main directory: {main_dir}")
            continue

        dataset_mapping[main_dir] = {}

        # Identify subfolders based on whether it's Testing or Training
        sub_dirs = expected_testing_subdirs if main_dir == "Testing" else expected_training_subdirs
        
        for sub_dir in sub_dirs:
            sub_path = os.path.join(main_path, sub_dir)
            if not os.path.exists(sub_path):
                print(f"Missing subdirectory: {sub_dir} in {main_dir}")
                continue

            dataset_mapping[main_dir][sub_dir] = {}

            # Traverse subfolders (e.g., specific datasets within Invivo, Phantoms, etc.)
            subfolders = [f for f in os.listdir(sub_path) if os.path.isdir(os.path.join(sub_path, f))]
            dataset_mapping[main_dir][sub_dir]["Total Folders"] = len(subfolders)

            for folder in subfolders:
                folder_path = os.path.join(sub_path, folder)
                
                # Count PA1 to PA7 images
                images = [f for f in os.listdir(folder_path) if f.startswith("PA") and f.endswith(".png")]
                
                dataset_mapping[main_dir][sub_dir][folder] = {
                    "Total Images": len(images),
                    "Expected 7 Images": "Yes" if len(images) == 7 else f"{len(images)} found"
                }

    return dataset_mapping

def save_mapping(data, filename):
    """Saves mapping data to JSON and CSV files in `src/config/`."""
    os.makedirs(CONFIG_DIR, exist_ok=True)

    # Save as JSON
    json_path = os.path.join(CONFIG_DIR, filename + ".json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Saved mapping to {json_path}")

    # Save as CSV
    csv_path = os.path.join(CONFIG_DIR, filename + ".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Main Directory", "Subdirectory", "Dataset Folder", "Total Images", "Expected 7 Images"])

        for main, subdirs in data.items():
            for sub, datasets in subdirs.items():
                total_folders = datasets.pop("Total Folders", None)
                writer.writerow([main, sub, "Total Folders", total_folders, ""])
                
                for dataset_folder, values in datasets.items():
                    writer.writerow([main, sub, dataset_folder, values["Total Images"], values["Expected 7 Images"]])

    print(f"Saved mapping to {csv_path}")

def explore_pa_experiment_data(root_dir):
    """
    Explores `data/pa_experiment_data/Training` to gather image statistics.

    :param root_dir: Path to the dataset.
    """
    print(f"\nExploring PA Experiment Data in: {root_dir}")

    expected_categories = ["KneeSlice1", "Phantoms", "SmallAnimal", "Transducers"]
    total_images = 0

    for category in expected_categories:
        category_path = os.path.join(root_dir, "Training", category)
        if not os.path.exists(category_path):
            print(f"Missing directory: {category_path}")
            continue

        subfolders = [f for f in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, f))]

        for subfolder in subfolders:
            subfolder_path = os.path.join(category_path, subfolder)
            image_files = sorted([f for f in os.listdir(subfolder_path) if f.startswith("PA") and f.endswith(".png")])

            num_images = len(image_files)
            total_images += num_images

            if num_images == 0:
                print(f"No images found in {subfolder_path}")
                continue

            # Load first image to check dimensions
            first_image = cv2.imread(os.path.join(subfolder_path, image_files[0]), cv2.IMREAD_GRAYSCALE)
            image_shape = first_image.shape if first_image is not None else "Unknown"
            image_type = first_image.dtype if first_image is not None else "Unknown"

            # Compute statistics
            all_pixels = np.array([cv2.imread(os.path.join(subfolder_path, f), cv2.IMREAD_GRAYSCALE).flatten() for f in image_files])
            min_val, max_val, mean_val = np.min(all_pixels), np.max(all_pixels), np.mean(all_pixels)

            print(f"\nKey: {category}/{subfolder}")
            print(f"   Shape: {image_shape}")
            print(f"   Type: {image_type}")
            print(f"   Min: {min_val:.4f}, Max: {max_val:.4f}, Mean: {mean_val:.4f}")
            print(f"   Number of images: {num_images}")
            print("-" * 50)

    print(f"\nTotal images in PA Experiment Data: {total_images}")


if __name__ == "__main__":
    dataset_root = "data/pa_experiment_data"

    # print("\n🔍 Mapping `data/pa_experiment_data/` structure...")
    # dataset_mapping = map_experiment_data_structure(dataset_root)
    # save_mapping(dataset_mapping, "experiment_data_mapping")
    explore_pa_experiment_data(dataset_root)