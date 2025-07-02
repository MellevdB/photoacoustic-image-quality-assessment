#!/bin/bash

# === CONFIGURATION ===
CSV_PATH="results/denoising_data/denoising_data_per_image_metrics_2025-05-20_13-29-51.csv"
METRIC="SSIM"
BASE_DIR="/home/mvanderbrugge/photoacoustic-image-quality-assessment"
JOB_DIR="$BASE_DIR/job_files"
OUT_DIR="$BASE_DIR/slurm_outputs"
SCRIPT_PATH="src/dl_model/cross_validate.py"
FC_UNITS_LIST=(64 128 256)
CONV_FILTER_LIST=("[8,16,32,64]" "[16,32,64,128]" "[32,64,128,256]")
DROPOUT_LIST=(0.0 0.3 0.5)

mkdir -p "$JOB_DIR"
mkdir -p "$OUT_DIR"

JOB_COUNTER=0
COMMANDS=()

# === LOOP OVER CONFIGURATIONS ===
for MODEL_TYPE in dropout batchnorm; do
  for LOSS_FN in mse huber antibias; do
    for OPTIMIZER in adam adamw; do
      for BATCH_SIZE in 16 32; do
        for FC_UNITS in "${FC_UNITS_LIST[@]}"; do
          for CONV_FILTERS in "${CONV_FILTER_LIST[@]}"; do

            if [ "$MODEL_TYPE" == "dropout" ]; then
              for DROPOUT in "${DROPOUT_LIST[@]}"; do
                NAME="cv_${MODEL_TYPE}_${LOSS_FN}_${OPTIMIZER}_bs${BATCH_SIZE}_fc${FC_UNITS}_drop${DROPOUT}_conv$(echo $CONV_FILTERS | tr -d '[],')"
                CMD="python $SCRIPT_PATH \
  --csv_path $CSV_PATH \
  --metric $METRIC \
  --model_type $MODEL_TYPE \
  --loss_fn $LOSS_FN \
  --optimizer $OPTIMIZER \
  --batch_size $BATCH_SIZE \
  --dropout $DROPOUT \
  --fc_units $FC_UNITS \
  --conv_filters \"$CONV_FILTERS\""
                COMMANDS+=("$CMD")
              done
            else
              NAME="cv_${MODEL_TYPE}_${LOSS_FN}_${OPTIMIZER}_bs${BATCH_SIZE}_fc${FC_UNITS}_conv$(echo $CONV_FILTERS | tr -d '[],')"
              CMD="python $SCRIPT_PATH \
  --csv_path $CSV_PATH \
  --metric $METRIC \
  --model_type $MODEL_TYPE \
  --loss_fn $LOSS_FN \
  --optimizer $OPTIMIZER \
  --batch_size $BATCH_SIZE \
  --fc_units $FC_UNITS \
  --conv_filters \"$CONV_FILTERS\""
              COMMANDS+=("$CMD")
            fi

            # === GROUP 4 commands per job ===
            while [ "${#COMMANDS[@]}" -ge 4 ]; do
              JOB_NAME="cv_pack_${JOB_COUNTER}"
              JOB_FILE="$JOB_DIR/$JOB_NAME.sh"
              echo "Submitting $JOB_NAME"

              cat <<EOF > "$JOB_FILE"
#!/bin/bash
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --job-name=$JOB_NAME
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=8:00:00
#SBATCH --output=$OUT_DIR/${JOB_NAME}_%A.out

module purge
module load 2023
module load Anaconda3/2023.07-2

cd $BASE_DIR
source activate photoacoustic-env
export PYTHONPATH=src

${COMMANDS[0]}
${COMMANDS[1]}
${COMMANDS[2]}
${COMMANDS[3]}
EOF

              sbatch "$JOB_FILE"
              COMMANDS=("${COMMANDS[@]:4}")  # Remove the first 4
              ((JOB_COUNTER++))
            done

          done
        done
      done
    done
  done
done

# Handle remaining (if not divisible by 4)
if [ "${#COMMANDS[@]}" -gt 0 ]; then
  JOB_NAME="cv_pack_${JOB_COUNTER}"
  JOB_FILE="$JOB_DIR/$JOB_NAME.sh"
  echo "Submitting leftover $JOB_NAME"

  {
    echo "#!/bin/bash"
    echo "#SBATCH --partition=gpu_a100"
    echo "#SBATCH --gpus=1"
    echo "#SBATCH --job-name=$JOB_NAME"
    echo "#SBATCH --ntasks=1"
    echo "#SBATCH --cpus-per-task=12"
    echo "#SBATCH --time=4:00:00"
    echo "#SBATCH --output=$OUT_DIR/${JOB_NAME}_%A.out"
    echo ""
    echo "module purge"
    echo "module load 2023"
    echo "module load Anaconda3/2023.07-2"
    echo ""
    echo "cd $BASE_DIR"
    echo "source activate photoacoustic-env"
    echo "export PYTHONPATH=src"
    echo ""
    for CMD in "${COMMANDS[@]}"; do
      echo "$CMD"
    done
  } > "$JOB_FILE"

  sbatch "$JOB_FILE"
fi