OADAT_DATA_DIR = "data/OADAT/"  
RESULTS_DIR = "results/oadat/"  


DATASETS = {
    "SCD": {
        "path": "SCD/SCD_RawBP-mini.h5",
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
            "multisegment": "SWFD/SWFD_multisegment_RawBP-mini.h5",
            "semicircle":   "SWFD/SWFD_semicircle_RawBP-mini.h5",
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
        "path": "MSFD/MSFD_multisegment_RawBP-mini.h5",
        "configs": {
            # limited view
            "lv128": ["li"],     
            "ss128": ["ms"],    
            # sparse view
            "ss64":  ["ms"],   
            "ss32":  ["ms"],    
        },
    },
}