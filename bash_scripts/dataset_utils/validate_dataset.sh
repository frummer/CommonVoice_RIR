#!/bin/bash 

# 

#SBATCH --job-name=validating-dataset-files #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --cpus-per-task=1   # Request 2 CPU cores
#SBATCH --mem=8G           # Request 16GB of RAM
#SBATCH --partition=cpu  #queue
#SBATCH --error=logs/logs_dataset_validation/logs_dataset_validate_length/extract.err
#SBATCH --output=logs/logs_dataset_validation/logs_dataset_validate_length/extract.out
#SBATCH --mail-user=afrumme1@jh.edu  #email for reporting

echo "Validating Datasets subdirs length"
module purge 
module load conda 

source activate /home/afrumme1/miniconda3/envs/common_voice_rir_3

/home/afrumme1/miniconda3/envs/common_voice_rir_3/bin/python \
    /home/afrumme1/CommonVoice_RIR/src/dataset_validation_utils/validate_dataset_subdirs_length.py  \
    --base_dir /export/fs05/afrumme1/babylon_datasets_1/english/train_01_02_2025_22_37_03_8_4_20_10_15_10/train