#!/bin/bash 

# 

#SBATCH --job-name=create-test-commonvoice-separation-dataset #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --cpus-per-task=2   # Request 2 CPU cores
#SBATCH --mem=16G           # Request 16GB of RAM
#SBATCH --partition=cpu  #queue
#SBATCH --error=logs_arabic_test/extract.err
#SBATCH --output=logs_arabic_test/extract.out
#SBATCH --mail-user=afrumme1@jh.edu  #email for reporting

echo "Creating CommonVoice Dataset - test split"
module purge 
module load conda 

source activate /home/afrumme1/miniconda3/envs/common_voice_rir_3
huggingface-cli login --token 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

/home/afrumme1/miniconda3/envs/common_voice_rir_3/bin/python \
    /home/afrumme1/CommonVoice_RIR/src/create_overlapped_dataset.py  \
    --config_path /home/afrumme1/CommonVoice_RIR/src/configs/create_overlapped_test_set_on_grid_config.json