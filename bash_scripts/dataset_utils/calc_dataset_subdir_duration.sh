#!/bin/bash 

#SBATCH --job-name=calculate-wav-duration  # Job name
#SBATCH --nodes=1  # Number of nodes requested
#SBATCH --cpus-per-task=8   # Request 1 CPU core
#SBATCH --mem=16G            # Request 8GB of RAM
#SBATCH --partition=cpu  # Queue
#SBATCH --error=/home/afrumme1/CommonVoice_RIR/logs/logs_dataset_validation/logs_LDC_validation_V2/logs_dataset_duration/validation/calculate_wav_duration.err
#SBATCH --output=/home/afrumme1/CommonVoice_RIR/logs/logs_dataset_validation/logs_LDC_validation_V2/logs_dataset_duration/validation/calculate_wav_duration.out
#SBATCH --mail-user=afrumme1@jh.edu  # Email for reporting

echo "Calculating total WAV duration..."
# Echo the current time
echo "Script started at: $(date)"
module purge 
module load conda 

# Ensure conda is properly initialized
conda activate /home/afrumme1/miniconda3/envs/common_voice_rir_3

# Define the WAV directory
WAV_DIR=/export/fs05/afrumme1/babylon_datasets_2/LDC2014S02/validation_13_02_2025_00_58_48_8_4_20_10_15_10/validation/mixture
WORKERS=$SLURM_CPUS_PER_TASK  # Use the allocated CPUs from SLURM

# Echo the arguments for verification
echo "Processing directory: $WAV_DIR"

# Run the Python script with the specified directory
/home/afrumme1/miniconda3/envs/common_voice_rir_3/bin/python \
    /home/afrumme1/CommonVoice_RIR/src/dataset_validation_utils/calc_dataaset_duration.py  \
    --directory $WAV_DIR \
    --workers $WORKERS

echo "WAV duration calculation completed."
