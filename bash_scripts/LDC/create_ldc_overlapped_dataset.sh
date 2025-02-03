#!/bin/bash 

# 

#SBATCH --job-name=create-test-en-commonvoice-separation-dataset #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --partition=cpu  #queue
#SBATCH --cpus-per-task=4   # Request 4 CPU cores
#SBATCH --mem=32G           # Request 32GB of RAM
#SBATCH --error=logs/logs_ldc_dataset/extract.err
#SBATCH --output=logs/logs_ldc_dataset/extract.out
#SBATCH --mail-user=afrumme1@jh.edu  #email for reporting

echo "Creating LDC 2-spk Mixtures dataset"
module purge 
module load conda 

source activate /home/afrumme1/miniconda3/envs/common_voice_rir_3
export PYTHONPATH=/home/afrumme1/CommonVoice_RIR:$PYTHONPATH

/home/afrumme1/miniconda3/envs/common_voice_rir_3/bin/python \
    /home/afrumme1/CommonVoice_RIR/src_ldc/create_overlapped_dataset.py  \
    --config_path /home/afrumme1/CommonVoice_RIR/src/configs/create_ldc_overlapped_dataset_set_config.json