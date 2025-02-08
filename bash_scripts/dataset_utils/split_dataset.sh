#!/bin/bash

#SBATCH --job-name=split-dataset        # Job name
#SBATCH --nodes=1                       # Number of nodes requested
#SBATCH --cpus-per-task=1              # Number of CPU cores
#SBATCH --mem=8G                        # Request 8GB of RAM
#SBATCH --partition=cpu                 # Queue (partition) name
#SBATCH --error=logs/logs_ldc/logs_split_dataset/split_dataset.err
#SBATCH --output=logs/logs_ldc/logs_split_dataset/split_dataset.out
#SBATCH --mail-user=afrumme1@jh.edu     # Email for reporting

echo "Starting dataset splitting job"
echo "Script started at: $(date)"

module purge
module load conda

# Define arguments
BASE_DIR="/export/fs05/afrumme1/babylon_datasets_1/LDC2014S02/dataset_02_02_2025_20_53_02_8_4_20_10_15_10/dataset"
OUTPUT_DIR="/export/fs05/afrumme1/babylon_datasets_1/LDC2014S02/v1_split"
SUB_DIRS="mixture compressed_mixture source1 source1_reverb source2 source2_reverb"
REFERENCE_DIR="mixture"
TRAIN_RATIO=0.8
VALIDATION_RATIO=0.1
SEED=42

# Echo the arguments for verification
echo "Base Directory:      $BASE_DIR"
echo "Output Directory:    $OUTPUT_DIR"
echo "Subdirectories:      $SUB_DIRS"
echo "Reference Directory: $REFERENCE_DIR"
echo "Train Ratio:         $TRAIN_RATIO"
echo "Validation Ratio:    $VALIDATION_RATIO"
echo "Seed:                $SEED"

# Activate your conda environment
source activate /home/afrumme1/miniconda3/envs/common_voice_rir_3

# Run the Python script
/home/afrumme1/miniconda3/envs/common_voice_rir_3/bin/python \
  /home/afrumme1/CommonVoice_RIR/src_ldc/split_dataset.py  \
  --base_dir $BASE_DIR \
  --output_dir $OUTPUT_DIR \
  --sub_dirs $SUB_DIRS \
  --reference_dir $REFERENCE_DIR \
  --train_ratio $TRAIN_RATIO \
  --validation_ratio $VALIDATION_RATIO \
  --seed $SEED

echo "Script finished at: $(date)"
