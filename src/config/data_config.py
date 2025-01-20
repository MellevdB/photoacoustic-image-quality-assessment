import os

# Base directories
DATA_DIR = "data/"
RESULTS_DIR = "results/"

DATASETS = {
    "SCD": {
        "path": os.path.join(DATA_DIR, "OADAT/SCD/SCD_RawBP-mini.h5"),
        "configs": {
            # limited view
            "lv128": ["li", "vc"],    
            "lv256": ["ms"],           
            # sparse view
            "ss128": ["vc", "ms"],     
            "ss64":  ["vc", "ms"],     
            "ss32":  ["vc", "ms"],     
        },
    },
    "SWFD": {
        "path": {
            "multisegment": os.path.join(DATA_DIR, "OADAT/SWFD/SWFD_multisegment_RawBP-mini.h5"),
            "semicircle":   os.path.join(DATA_DIR, "OADAT/SWFD/SWFD_semicircle_RawBP-mini.h5"),
        },
        "configs": {
            # limited view
            "lv128": ["li", "sc"],    
            # sparse view
            "ss128": ["sc", "ms"],    
            "ss64":  ["sc", "ms"],    
            "ss32":  ["sc", "ms"],    
        },
    },
    "MSFD": {
        "path": os.path.join(DATA_DIR, "OADAT/MSFD/MSFD_multisegment_RawBP-mini.h5"),
        "configs": {
            # limited view
            "lv128": ["li"],     
            "ss128": ["ms"],    
            # sparse view
            "ss64":  ["ms"],   
            "ss32":  ["ms"],    
        },
    },
    "mice": {
        "path": os.path.join(DATA_DIR, "mice/"),
        "configs": {
            "sparse4":  ["sparse4_recon_all"],
            "sparse8":  ["sparse8_recon_all"],
            "sparse16": ["sparse16_recon_all"],
            "sparse32": ["sparse32_recon_all"],
            "sparse64": ["sparse64_recon_all"],
            "sparse128": ["sparse128_recon_all"],
            "sparse256": ["sparse256_recon_all"],
            "full":     ["full_recon_all"],  # Ground truth
        },
        "ground_truth": "full_recon_all",
    },
    "phantom": {
        "path": os.path.join(DATA_DIR, "phantom/"),
        "configs": {
            "sparse8":   ["BP_phantom_8"],
            "sparse16":  ["BP_phantom_16"],
            "sparse32":  ["BP_phantom_32"],
            "sparse64":  ["BP_phantom_64"],
            "sparse128": ["BP_phantom_128"],
            "full":      ["BP_phantom_GT"],  # Ground truth
        },
        "ground_truth": "BP_phantom_GT",
    },
    "v_phantom": {
        "path": os.path.join(DATA_DIR, "v_phantom/"),
        "configs": {
            "sparse8":   ["v_phantom_8"],
            "sparse16":  ["v_phantom_16"],
            "sparse32":  ["v_phantom_32"],
            "sparse64":  ["v_phantom_64"],
            "sparse128": ["v_phantom_128"],
            "full":      ["v_phantom_gt"],  # Ground truth
        },
        "ground_truth": "v_phantom_gt",
    },
}

# Ensure results directories exist
for dataset in DATASETS:
    os.makedirs(os.path.join(RESULTS_DIR, dataset), exist_ok=True)