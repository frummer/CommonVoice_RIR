#!/bin/bash 

# 

#SBATCH --job-name=separation #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --partition=cpu  #queue
#SBATCH --error=logs/extract.err
#SBATCH --output=logs/extract.out
#SBATCH --mail-user=afrumme1@jh.edu  #email for reporting

echo "Creating CommonVoice Dataset"
module purge 
module load conda 

source activate /home/afrumme1/miniconda3/envs/commonvoice_rir_2
huggingface-cli login --token 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

/home/afrumme1/miniconda3/envs/commonvoice_rir_2/bin/python /home/afrumme1/CommonVoice_RIR/src/create_overlapped_dataset.py