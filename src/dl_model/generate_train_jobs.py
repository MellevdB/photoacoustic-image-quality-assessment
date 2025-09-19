import os
from pathlib import Path

models = ["best_model", "IQDCNN", "EfficientNetIQA"]
metrics = [
    'SSIM', 'GMSD_norm', 'HAARPSI', 'IWSSIM', 'S3IM',
    ['SSIM', 'GMSD_norm'],
    ['SSIM', 'HAARPSI'],
    ['GMSD_norm', 'HAARPSI'],
    ['SSIM', 'GMSD_norm', 'HAARPSI', 'S3IM', 'IWSSIM']
]

job_dir = Path("train_jobs")
job_dir.mkdir(exist_ok=True)

submit_jobs = True  # Set to False if you only want to generate without submitting

for model in models:
    for metric in metrics:
        # Format metric name safely
        if isinstance(metric, str):
            metric_label = metric
            metric_arg = f"'{metric}'"
        else:
            metric_label = "_".join(metric)
            metric_arg = str(metric).replace("'", '"')  # For JSON-like list syntax

        job_name = f"train_{metric_label}_{model}"
        script_path = job_dir / f"{job_name}.sh"

        with open(script_path, "w") as f:
            f.write(f"""#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name={job_name}
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mellevdbrugge@gmail.com
#SBATCH --output=slurm_outputs/{job_name}_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

cd /home/mvanderbrugge/photoacoustic-image-quality-assessment
source activate photoacoustic-env
export PYTHONPATH=src

echo "Running on $(hostname)"
nvidia-smi
free -h

echo "Starting training: model={model}, metric={metric_label}"
srun python -c \\
"from dl_model.train import train_model; \\
train_model(data_dir='results', batch_size=16, learning_rate=1e-4, num_epochs=100, device='cuda', \\
save_path='models/{model}/{metric_label}/best_model.pth', \\
target_metric={metric_arg}, until_convergence=True, patience=10, dropout_rate={0.0 if model != 'IQDCNN' else 0.3}, \\
num_fc_units={128 if model != 'IQDCNN' else 1024}, conv_filters={[32,64,128,256] if model != 'IQDCNN' else [32,32,32,32]}, \\
model_variant='{('dropout' if model == 'best_model' else 'iqdcnn' if model == 'IQDCNN' else 'efficientnet') + ('_multi' if isinstance(metric, list) else '')}', \\
loss_fn='l1', optimizer='adam')"
""")

        print(f"Created job script: {script_path}")

        if submit_jobs:
            os.system(f"sbatch {script_path}")