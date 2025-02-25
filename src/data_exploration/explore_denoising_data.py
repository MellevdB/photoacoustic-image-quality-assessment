import os
import json
import csv

CONFIG_DIR = "src/config"

def map_denoising_data_structure(root_dir):
    """
    Maps the dataset structure in `data/denoising_data/`, ensuring expected quality levels and ground truth exist.

    :param root_dir: Path to dataset directory.
    :return: Dictionary containing structure mapping.
    """
    dataset_mapping = {}

    # Expected subdirectories
    expected_main_dirs = ["drive", "experimental", "nne"]
    expected_sub_dirs = ["test", "train", "validation"]
    expected_quality_levels = ["10db", "20db", "30db", "40db", "50db", "ground_truth"]

    # Walk through the main directories
    for main_dir in expected_main_dirs:
        main_path = os.path.join(root_dir, main_dir)
        if not os.path.exists(main_path):
            print(f"⚠️ Missing main directory: {main_dir}")
            continue

        dataset_mapping[main_dir] = {}

        # Handling "drive" and "nne" (similar structure, different file formats)
        if main_dir in ["drive", "nne"]:
            file_extension = ".jpg" if main_dir == "drive" else ".png"

            for sub_dir in expected_sub_dirs:
                sub_path = os.path.join(main_path, sub_dir)
                if not os.path.exists(sub_path):
                    print(f"⚠️ Missing subdirectory: {sub_dir} in {main_dir}")
                    continue

                dataset_mapping[main_dir][sub_dir] = {}

                # Scan for quality level folders
                for quality in expected_quality_levels:
                    quality_path = os.path.join(sub_path, quality)
                    if os.path.exists(quality_path):
                        files = [f for f in os.listdir(quality_path) if f.endswith(file_extension)]
                        dataset_mapping[main_dir][sub_dir][quality] = len(files)
                    else:
                        dataset_mapping[main_dir][sub_dir][quality] = "MISSING"

                # Scan for YAML files
                yaml_files = [f for f in os.listdir(sub_path) if f.endswith(".yaml")]
                dataset_mapping[main_dir][sub_dir]["yaml_files"] = yaml_files

        # Handling "experimental" (different structure, uses PNG)
        elif main_dir == "experimental":
            dataset_mapping[main_dir] = {}
            for exp_sub in ["ground_truth", "input"]:
                exp_sub_path = os.path.join(main_path, exp_sub)
                if os.path.exists(exp_sub_path):
                    files = [f for f in os.listdir(exp_sub_path) if f.endswith(".png")]
                    dataset_mapping[main_dir][exp_sub] = len(files)
                else:
                    dataset_mapping[main_dir][exp_sub] = "MISSING"

            # Check for `data.yaml`
            yaml_path = os.path.join(main_path, "data.yaml")
            dataset_mapping[main_dir]["data.yaml"] = "EXISTS" if os.path.exists(yaml_path) else "MISSING"

    return dataset_mapping

def save_mapping(data, filename):
    """Saves mapping data to JSON and CSV files in `src/config/`."""
    os.makedirs(CONFIG_DIR, exist_ok=True)

    # Save as JSON
    json_path = os.path.join(CONFIG_DIR, filename + ".json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"✅ Saved mapping to {json_path}")

    # Save as CSV
    csv_path = os.path.join(CONFIG_DIR, filename + ".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Main Directory", "Subdirectory", "Category", "Number of Files"])

        for main, subdirs in data.items():
            for sub, categories in subdirs.items():
                if isinstance(categories, dict):
                    for cat, value in categories.items():
                        writer.writerow([main, sub, cat, value])
                else:
                    writer.writerow([main, sub, "Total Files", categories])

    print(f"✅ Saved mapping to {csv_path}")

if __name__ == "__main__":
    data_root = "data/denoising_data"

    print("\n🔍 Mapping `data/denoising_data/` structure...")
    data_mapping = map_denoising_data_structure(data_root)
    save_mapping(data_mapping, "denoising_data_mapping")