import os
import pandas as pd

RESULTS_DIR = "results/eval_model/cross_validate"
rows = []

# Track the global best config
best_loss = float('inf')
best_file = None
best_entry = None

# Recursively walk through all subdirectories
for root, _, files in os.walk(RESULTS_DIR):
    for file in files:
        if file.endswith(".txt"):
            filepath = os.path.join(root, file)
            with open(filepath, 'r') as f:
                content = f.read()
                entry = {}

                for line in content.strip().splitlines():
                    if ":" not in line:
                        continue
                    key, value = line.split(":", 1)
                    key = key.strip()
                    value = value.strip()

                    # Convert known fields
                    if key == "Dropout":
                        entry["Dropout"] = float(value)
                    elif key == "FC Units":
                        entry["FC Units"] = int(value)
                    elif key == "Batch Size":
                        entry["Batch Size"] = int(value)
                    elif key == "LR":
                        entry["LR"] = float(value)
                    elif key == "Epochs":
                        entry["Epochs"] = int(value)
                    elif key == "Folds":
                        entry["Folds"] = int(value)
                    elif key == "Mean Val Loss":
                        entry["Mean Val Loss"] = float(value)
                    elif key == "Fold losses":
                        entry["Fold losses"] = eval(value)
                    elif key == "Conv Filters":
                        entry["Conv Filters"] = value
                    else:
                        entry[key] = value

                entry["File"] = filepath
                rows.append(entry)

                # Update best configuration
                if "Mean Val Loss" in entry and entry["Mean Val Loss"] < best_loss:
                    best_loss = entry["Mean Val Loss"]
                    best_file = filepath
                    best_entry = entry.copy()

# Print best configuration
if best_entry is not None:
    print("\nBest configuration found:")
    print(f"  → File path     : {best_file}")
    print(f"  → Mean Val Loss : {best_loss:.6f}")
    print("  → Config        :")
    for k in ["Model", "Loss", "Optimizer", "Dropout", "FC Units", "Conv Filters", "Batch Size", "LR"]:
        print(f"     {k:<12}: {best_entry.get(k)}")
else:
    print("No valid configuration files found.")

# Save summary CSV
df = pd.DataFrame(rows)
df_sorted = df.sort_values(by="Mean Val Loss")
print("\nTop 5 configurations by lowest Mean Val Loss:")
print(df_sorted[[
    "Mean Val Loss", "Model", "Loss", "Optimizer", "Dropout",
    "FC Units", "Conv Filters", "Batch Size", "LR", "Epochs", "File"
]].head(5))
df_sorted.to_csv("results/eval_model/cross_validate_summary.csv", index=False)