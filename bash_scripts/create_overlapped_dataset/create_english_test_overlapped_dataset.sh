#!/bin/bash 

# 

#SBATCH --job-name=create-test-en-commonvoice-separation-dataset #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --partition=cpu  #queue
#SBATCH --cpus-per-task=1   # Request 4 CPU cores
#SBATCH --mem=8G           # Request 32GB of RAM
#SBATCH --error=logs/logs_english_common_voice/logs_english_test/extract.err
#SBATCH --output=logs/logs_english_common_voice/logs_english_test/extract.out
#SBATCH --mail-user=afrumme1@jh.edu  #email for reporting

echo "Creating CommonVoice Dataset - test split"
module purge 
module load conda 

source activate /home/afrumme1/miniconda3/envs/common_voice_rir_3
huggingface-cli login --token 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
export PYTHONPATH=/home/afrumme1/CommonVoice_RIR:$PYTHONPATH

/home/afrumme1/miniconda3/envs/common_voice_rir_3/bin/python \
    /home/afrumme1/CommonVoice_RIR/src/create_overlapped_dataset.py  \
    --config_path /home/afrumme1/CommonVoice_RIR/src/configs/create_overlapped_english_test_set_on_grid_config.json