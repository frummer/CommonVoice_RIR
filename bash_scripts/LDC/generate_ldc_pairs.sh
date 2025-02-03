#!/bin/bash 

# 

#SBATCH --job-name=generating-pairs-from-LDC-Dataset #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --cpus-per-task=1   # Request 2 CPU cores
#SBATCH --mem=8G           # Request 16GB of RAM
#SBATCH --partition=cpu  #queue
#SBATCH --error=logs_ldc_pairing/extract.err
#SBATCH --output=logs_ldc_pairing/extract.out
#SBATCH --mail-user=afrumme1@jh.edu  #email for reporting

echo "generating pairs from LDC Dataset"
module purge 
module load conda 

source activate /home/afrumme1/miniconda3/envs/common_voice_rir_3

/home/afrumme1/miniconda3/envs/common_voice_rir_3/bin/python \
    /home/afrumme1/CommonVoice_RIR/src_ldc/generate_speakers_pairs.py  \
    --input_csv /home/afrumme1/CommonVoice_RIR/csv_output/utterance_mapping.csv \
    --output_csv /home/afrumme1/CommonVoice_RIR/csv_output/output_pairs.csv \
    --utterance_count_csv /home/afrumme1/CommonVoice_RIR/csv_output/output_utterance_count.csv \
    --speaker_pair_count_csv /home/afrumme1/CommonVoice_RIR/csv_output/output_pair_count.csv \

