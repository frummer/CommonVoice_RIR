#!/bin/bash 

# 

#SBATCH --job-name=combine_and_split #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --partition=cpu  #queue
#SBATCH --error=logs_other/extract.err
#SBATCH --output=logs_other/extract.out
#SBATCH --mail-user=afrumme1@jh.edu  #email for reporting

echo "Creating CommonVoice Dataset"
module purge 
module load conda 

source activate /home/afrumme1/miniconda3/envs/commonvoice_rir_2

/home/afrumme1/miniconda3/envs/commonvoice_rir_2/bin/python /home/afrumme1/CommonVoice_RIR/src/common_voice_utils/combine_and_split.py