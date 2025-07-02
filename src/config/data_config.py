import os

# Base directories
DATA_DIR = "data/"
RESULTS_DIR = "results/"

DATASETS = {
    # SCD dataset
    "SCD": {
        "path": os.path.join(DATA_DIR, "OADAT/SCD/SCD_RawBP-mini.h5"),
        "configs": {
            # Limited view
            "lv128": ["vc,lv128_BP"],
            # Sparse view
            "ss128": ["vc,ss128_BP"],
            "ss64": ["vc,ss64_BP"],
            "ss32": ["vc,ss32_BP"],
            # Additional configurations
            "linear": ["linear_BP"],
            "ms_ss128": ["ms,ss128_BP"],
            "ms_ss64": ["ms,ss64_BP"],
            "ms_ss32": ["ms,ss32_BP"],
        },
        "ground_truth": {
            "vc": "vc_BP",  # Ground truth for vc configurations
            "linear": "linear_BP",  # Ground truth for linear configuration
            "ms": "ms_BP",  # Ground truth for ms configurations
        },
    },

    # SWFD dataset
    "SWFD": {
        "path": {
            "multisegment": os.path.join(DATA_DIR, "OADAT/SWFD/SWFD_multisegment_RawBP-mini.h5"),
            "semicircle": os.path.join(DATA_DIR, "OADAT/SWFD/SWFD_semicircle_RawBP-mini.h5"),
        },
        "configs": {
            # Multisegment configurations
            "ms_lv128": ["linear_BP"],
            "ms_ss128": ["ms,ss128_BP"],
            "ms_ss64": ["ms,ss64_BP"],
            "ms_ss32": ["ms,ss32_BP"],
            # Semicircle configurations
            "sc_lv128": ["sc,lv128_BP"],
            "sc_ss128": ["sc,ss128_BP"],
            "sc_ss64": ["sc,ss64_BP"],
            "sc_ss32": ["sc,ss32_BP"],
        },
        "ground_truth": {
            "multisegment": "ms_BP",  # Ground truth for multisegment configurations
            "semicircle": "sc_BP",  # Ground truth for semicircle configurations
        },
    },

    # MSFD dataset
    "MSFD": {
        "path": os.path.join(DATA_DIR, "OADAT/MSFD/MSFD_multisegment_RawBP-mini.h5"),
        "configs": {
            # Sparse view by wavelengths
            "ss32": ["ms,ss32_BP_w700", "ms,ss32_BP_w730", "ms,ss32_BP_w760", "ms,ss32_BP_w780", "ms,ss32_BP_w800", "ms,ss32_BP_w850"],
            "ss64": ["ms,ss64_BP_w700", "ms,ss64_BP_w730", "ms,ss64_BP_w760", "ms,ss64_BP_w780", "ms,ss64_BP_w800", "ms,ss64_BP_w850"],
            "ss128": ["ms,ss128_BP_w700", "ms,ss128_BP_w730", "ms,ss128_BP_w760", "ms,ss128_BP_w780", "ms,ss128_BP_w800", "ms,ss128_BP_w850"],
        },
        "ground_truth": {
            "wavelengths": {
                700: "ms_BP_w700",
                730: "ms_BP_w730",
                760: "ms_BP_w760",
                780: "ms_BP_w780",
                800: "ms_BP_w800",
                850: "ms_BP_w850",
            },
        },
    },

    # Full SCD
    "SCD_full": {
        "path": "/projects/prjs1596/photoacoustic/data/OADAT-full/SCD_RawBP.h5",
        "configs": {
            "lv128": ["vc,lv128_BP"],
            "ss128": ["vc,ss128_BP"],
            "ss64": ["vc,ss64_BP"],
            "ss32": ["vc,ss32_BP"],
            "linear": ["linear_BP"],
        },
        "ground_truth": {
            "vc": "vc_BP",
            "linear": "linear_BP",
        },
    },

    "SCD_ms_lv128_full": {
        "path": {
            "recon": "/projects/prjs1596/photoacoustic/data/OADAT-full/SCD_multisegment_ss_RawBP.h5",
            "gt": "/projects/prjs1596/photoacoustic/data/OADAT-full/SCD_RawBP.h5"
        },
        "configs": {
            "ms_lv128": ["ms,lv128_BP"],
        },
        "ground_truth": {
            "ms": "ms_BP",
        },
    },

    "SCD_ms_ss128_full": {
        "path": {
            "recon": "/projects/prjs1596/photoacoustic/data/OADAT-full/SCD_multisegment_ss_RawBP.h5",
            "gt": "/projects/prjs1596/photoacoustic/data/OADAT-full/SCD_RawBP.h5"
        },
        "configs": {
            "ms_ss128": ["ms,ss128_BP"],
        },
        "ground_truth": {
            "ms": "ms_BP",
        },
    },

    "SCD_ms_ss64_full": {
        "path": {
            "recon": "/projects/prjs1596/photoacoustic/data/OADAT-full/SCD_multisegment_ss_RawBP.h5",
            "gt": "/projects/prjs1596/photoacoustic/data/OADAT-full/SCD_RawBP.h5"
        },
        "configs": {
            "ms_ss64": ["ms,ss64_BP"],
        },
        "ground_truth": {
            "ms": "ms_BP",
        },
    },

    "SCD_ms_ss32_full": {
        "path": {
            "recon": "/projects/prjs1596/photoacoustic/data/OADAT-full/SCD_multisegment_ss_RawBP.h5",
            "gt": "/projects/prjs1596/photoacoustic/data/OADAT-full/SCD_RawBP.h5"
        },
        "configs": {
            "ms_ss32": ["ms,ss32_BP"],
        },
        "ground_truth": {
            "ms": "ms_BP",
        },
    },

    "SWFD_multisegment_ss_full": {
        "path": "/projects/prjs1596/photoacoustic/data/OADAT-full/SWFD_multisegment_ss_RawBP.h5",
        "configs": {
            "ms_ss128": ["ms,ss128_BP"],
            "ms_ss64": ["ms,ss64_BP"],
            "ms_ss32": ["ms,ss32_BP"],
        },
        "ground_truth": {
            "ms": "ms_BP",
        },
    },

    "SWFD_semicircle_full": {
        "path": "/projects/prjs1596/photoacoustic/data/OADAT-full/SWFD_semicircle_RawBP.h5",
        "configs": {
            "sc_ss128": ["sc,ss128_BP"],
            "sc_ss64": ["sc,ss64_BP"],
            "sc_ss32": ["sc,ss32_BP"],
            "sc_lv128": ["sc,lv128_BP"],
        },
        "ground_truth": {
            "sc": "sc_BP",
        },
    },

    "MSFD_full_w700": {
        "path": "/projects/prjs1596/photoacoustic/data/OADAT-full/MSFD_multisegment_ss_RawBP.h5",
        "configs": {
            "ms_ss32": ["ms,ss32_BP_w700"],
            "ms_ss64": ["ms,ss64_BP_w700"],
            "ms_ss128": ["ms,ss128_BP_w700"],
        },
        "ground_truth": {
            "wavelengths": {
                "700": "ms_BP_w700"
            }
        },
    },

    "MSFD_full_w730": {
        "path": "/projects/prjs1596/photoacoustic/data/OADAT-full/MSFD_multisegment_ss_RawBP.h5",
        "configs": {
            "ms_ss32": ["ms,ss32_BP_w730"],
            "ms_ss64": ["ms,ss64_BP_w730"],
            "ms_ss128": ["ms,ss128_BP_w730"],
        },
        "ground_truth": {
            "wavelengths": {
                "730": "ms_BP_w730"
            }
        },
    },

    "MSFD_full_w760": {
        "path": "/projects/prjs1596/photoacoustic/data/OADAT-full/MSFD_multisegment_ss_RawBP.h5",
        "configs": {
            "ms_ss32": ["ms,ss32_BP_w760"],
            "ms_ss64": ["ms,ss64_BP_w760"],
            "ms_ss128": ["ms,ss128_BP_w760"],
        },
        "ground_truth": {
            "wavelengths": {
                "760": "ms_BP_w760"
            }
        },
    },

    "MSFD_full_w780": {
        "path": "/projects/prjs1596/photoacoustic/data/OADAT-full/MSFD_multisegment_ss_RawBP.h5",
        "configs": {
            "ms_ss32": ["ms,ss32_BP_w780"],
            "ms_ss64": ["ms,ss64_BP_w780"],
            "ms_ss128": ["ms,ss128_BP_w780"],
        },
        "ground_truth": {
            "wavelengths": {
                "780": "ms_BP_w780"
            }
        },
    },

    "MSFD_full_w800": {
        "path": "/projects/prjs1596/photoacoustic/data/OADAT-full/MSFD_multisegment_ss_RawBP.h5",
        "configs": {
            "ms_ss32": ["ms,ss32_BP_w800"],
            "ms_ss64": ["ms,ss64_BP_w800"],
            "ms_ss128": ["ms,ss128_BP_w800"],
        },
        "ground_truth": {
            "wavelengths": {
                "800": "ms_BP_w800"
            }
        },
    },

    "MSFD_full_w850": {
        "path": "/projects/prjs1596/photoacoustic/data/OADAT-full/MSFD_multisegment_ss_RawBP.h5",
        "configs": {
            "ms_ss32": ["ms,ss32_BP_w850"],
            "ms_ss64": ["ms,ss64_BP_w850"],
            "ms_ss128": ["ms,ss128_BP_w850"],
        },
        "ground_truth": {
            "wavelengths": {
                "850": "ms_BP_w850"
            }
        },
    },

    # Mice dataset
    "mice": {
        "path": os.path.join(DATA_DIR, "mice/"),
        "configs": {
            "sparse4": ["sparse4_recon_all"],
            "sparse8": ["sparse8_recon_all"],
            "sparse16": ["sparse16_recon_all"],
            "sparse32": ["sparse32_recon_all"],
            "sparse64": ["sparse64_recon_all"],
            "sparse128": ["sparse128_recon_all"],
            "sparse256": ["sparse256_recon_all"],
            "full": ["full_recon_all"],  # Ground truth
        },
        "ground_truth": "full_recon_all",
    },

    # Phantom dataset
    "phantom": {
        "path": os.path.join(DATA_DIR, "phantom/"),
        "configs": {
            "sparse8": ["BP_phantom_8"],
            "sparse16": ["BP_phantom_16"],
            "sparse32": ["BP_phantom_32"],
            "sparse64": ["BP_phantom_64"],
            "sparse128": ["BP_phantom_128"],
            "full": ["BP_phantom_GT"],  # Ground truth
        },
        "ground_truth": "BP_phantom_GT",
    },

    # Virtual Phantom dataset
    "v_phantom": {
        "path": os.path.join(DATA_DIR, "v_phantom/"),
        "configs": {
            "sparse8": ["v_phantom_8"],
            "sparse16": ["v_phantom_16"],
            "sparse32": ["v_phantom_32"],
            "sparse64": ["v_phantom_64"],
            "sparse128": ["v_phantom_128"],
            "full": ["v_phantom_gt"],  # Ground truth
        },
        "ground_truth": "v_phantom_gt",
    },

    # Simulated PA Data
    "denoising_data": {
        "path": os.path.join(DATA_DIR, "denoising_data/"),
        "subsets": ["train", "test", "validation"],
        "categories": ["10db", "20db", "30db", "40db", "50db", "ground_truth"],
        "ground_truth": "ground_truth",
        "image_extension": {
            "drive": ".jpg",  # DRIVE dataset uses JPG
            "nne": ".png",  # NNE dataset uses PNG
        }
    },

    # Experimental Phantom Data
    "pa_experiment_data": {
        "path": os.path.join(DATA_DIR, "pa_experiment_data/"),
        "subsets": ["Training", "Testing"],
        "training_categories": ["KneeSlice1", "Phantoms", "SmallAnimal", "Transducers"],
        "testing_categories": ["Invivo", "Phantoms"],
        "num_configs": 7,  # PA1.png to PA7.png, where PA1.png is ground truth
        "ground_truth": "PA1.png",
        "image_extension": ".png",
    },

    "zenodo": {
        "path": "data/zenodo/normalised_images",
        "reference": "reference",
        "algorithms": "algorithms",
        "categories": ["0", "1", "2"],  # Corresponding to reconstruction methods
        "ground_truth": "reference"
    },

    "varied_split": {
    "path": "data/VARIED SPLIT V3 CURRENT"
}
}

# Ensure results directories exist
for dataset in DATASETS:
    os.makedirs(os.path.join(RESULTS_DIR, dataset), exist_ok=True)