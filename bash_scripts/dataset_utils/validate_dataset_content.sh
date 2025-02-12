#!/bin/bash 

#SBATCH --job-name=check-file-consistency  # Job name
#SBATCH --nodes=1  # Number of nodes requested
#SBATCH --cpus-per-task=1   # Request 1 CPU core
#SBATCH --mem=8G            # Request 8GB of RAM
#SBATCH --partition=cpu  # Queue
#SBATCH --error=logs/logs_dataset_validation/logs_arabic_commmon_voice_validation/logs_dataset_validate_content/test/check_files.err
#SBATCH --output=logs/logs_dataset_validation/logs_arabic_commmon_voice_validation/logs_dataset_validate_content/test/check_files.out
#SBATCH --mail-user=afrumme1@jh.edu  # Email for reporting

echo "Starting file consistency check..."
# Echo the current time
echo "Script started at: $(date)"
module purge 
module load conda 

# Ensure conda is properly initialized
conda activate /home/afrumme1/miniconda3/envs/common_voice_rir_3

# Define arguments
BASE_DIR=/export/fs05/afrumme1/babylon_datasets_1/arabic/test_01_02_2025_22_25_47_8_4_20_10_15_10/test/mixture
PREFIXES=comp_

# Echo the arguments for verification
echo "Base Directory: $BASE_DIR"
echo "Prefixes: $PREFIXES"

# Run the Python script with arguments
/home/afrumme1/miniconda3/envs/common_voice_rir_3/bin/python \
    /home/afrumme1/CommonVoice_RIR/src/dataset_validation_utils/validate_datasets_subdirs_content.py  \
    --base_dir $BASE_DIR \
    --prefixes $PREFIXES

echo "File consistency check completed."
