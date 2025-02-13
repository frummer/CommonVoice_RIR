#!/bin/bash 

# 

#SBATCH --job-name=validating-dataset-files #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --cpus-per-task=1   # Request 2 CPU cores
#SBATCH --mem=8G           # Request 16GB of RAM
#SBATCH --partition=cpu  #queue
#SBATCH --error=/home/afrumme1/CommonVoice_RIR/logs/logs_dataset_validation/logs_LDC_validation_V2/validate_dataset_subdirs_length/validation/extract.err
#SBATCH --output=/home/afrumme1/CommonVoice_RIR/logs/logs_dataset_validation/logs_LDC_validation_V2/validate_dataset_subdirs_length/validation/extract.out
#SBATCH --mail-user=afrumme1@jh.edu  #email for reporting
#SBATCH --mail-type=END,FAIL,BEGIN
echo "Validating Datasets subdirs length"
# Echo the current time
echo "Script started at: $(date)"
module purge 
module load conda 

# Define arguments
BASE_DIR=/export/fs05/afrumme1/babylon_datasets_2/LDC2014S02/validation_13_02_2025_00_58_48_8_4_20_10_15_10/validation
# Echo the arguments for verification
echo "Base Directory: $BASE_DIR"

source activate /home/afrumme1/miniconda3/envs/common_voice_rir_3

/home/afrumme1/miniconda3/envs/common_voice_rir_3/bin/python \
    /home/afrumme1/CommonVoice_RIR/src/dataset_validation_utils/validate_dataset_subdirs_length.py  \
    --base_dir $BASE_DIR 