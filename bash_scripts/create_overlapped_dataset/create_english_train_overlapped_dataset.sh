#!/bin/bash 

# 

#SBATCH --job-name=create-train-commonvoice-separation-dataset #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --partition=cpu  #queue
#SBATCH --cpus-per-task=4   # Request 4 CPU cores
#SBATCH --mem=32G           # Request 32GB of RAM
#SBATCH --error=logs_english_train/extract.err
#SBATCH --output=logs_english_train/extract.out
#SBATCH --mail-user=afrumme1@jh.edu  #email for reporting

echo "Creating CommonVoice Dataset - train split"
module purge 
module load conda 

source activate /home/afrumme1/miniconda3/envs/common_voice_rir_3
huggingface-cli login --token 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

/home/afrumme1/miniconda3/envs/common_voice_rir_3/bin/python \
    /home/afrumme1/CommonVoice_RIR/src/create_overlapped_dataset.py  \
    --config_path /home/afrumme1/CommonVoice_RIR/src/configs/create_overlapped_english_train_set_on_grid_config.json