# Denoising configurations
DENOISING_CONFIGS = ['10db', '20db', '30db', '40db', '50db']

# Mice dataset configurations
MICE_CONFIGS = [
    'sparse4_recon_all',
    'sparse8_recon_all',
    'sparse16_recon_all',
    'sparse32_recon_all',
    'sparse64_recon_all',
    'sparse128_recon_all',
    'sparse256_recon_all'
]

# Phantom dataset configurations
PHANTOM_CONFIGS = [
    'BP_phantom_8',
    'BP_phantom_16',
    'BP_phantom_32',
    'BP_phantom_64',
    'BP_phantom_128'
]

# V-Phantom dataset configurations
V_PHANTOM_CONFIGS = [
    'v_phantom_8',
    'v_phantom_16',
    'v_phantom_32',
    'v_phantom_64',
    'v_phantom_128'
]

# MSFD configurations
MSFD_CONFIGS = [
    'ms,ss32_BP_w760',
    'ms,ss64_BP_w760',
    'ms,ss128_BP_w760'
]

# SCD configurations
SCD_CONFIGS = {
    'virtual_circle': [
        'vc,ss32_BP',
        'vc,ss64_BP',
        'vc,ss128_BP'
    ],
    'multi_segment': [
        'ms,ss32_BP',
        'ms,ss64_BP',
        'ms,ss128_BP'
    ]
}

# SWFD configurations
SWFD_CONFIGS = {
    'multi_segment': [
        'ms,ss32_BP',
        'ms,ss64_BP',
        'ms,ss128_BP'
    ],
    'semi_circle': [
        'sc,ss32_BP',
        'sc,ss64_BP',
        'sc,ss128_BP'
    ]
}

# PA Experiment configurations
PA_EXPERIMENT_CONFIGS = {
    'KneeSlice1': ['PA2', 'PA3', 'PA4', 'PA5', 'PA6', 'PA7'],
    'Phantoms': ['PA2', 'PA3', 'PA4', 'PA5', 'PA6', 'PA7'],
    'Transducers': ['PA2', 'PA3', 'PA4', 'PA5', 'PA6', 'PA7'],
    'SmallAnimal': ['PA2', 'PA3', 'PA4', 'PA5', 'PA6', 'PA7']
}

# Dictionary containing all configurations
DATASET_CONFIGS = {
    'denoising': DENOISING_CONFIGS,
    'mice': MICE_CONFIGS,
    'phantom': PHANTOM_CONFIGS,
    'v_phantom': V_PHANTOM_CONFIGS,
    'msfd': MSFD_CONFIGS,
    'scd': SCD_CONFIGS,
    'swfd': SWFD_CONFIGS,
    'pa_experiment': PA_EXPERIMENT_CONFIGS
} 