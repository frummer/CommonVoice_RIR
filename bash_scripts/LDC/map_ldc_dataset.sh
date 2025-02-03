#!/bin/bash 

# 

#SBATCH --job-name=map-lddc-dataset #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --cpus-per-task=1   # Request 2 CPU cores
#SBATCH --mem=8G           # Request 16GB of RAM
#SBATCH --partition=cpu  #queue
#SBATCH --error=logs_ldc_mapping/extract.err
#SBATCH --output=logs_ldc_mapping/extract.out
#SBATCH --mail-user=afrumme1@jh.edu  #email for reporting

echo "Mapping LDC Dataset"
module purge 
module load conda 

source activate /home/afrumme1/miniconda3/envs/common_voice_rir_3

/home/afrumme1/miniconda3/envs/common_voice_rir_3/bin/python \
    /home/afrumme1/CommonVoice_RIR/src_ldc/scan_and_create_ldc_csv.py  \
    --dataset_root /export/corpora5/LDC/LDC2014S02/data \
    --output_dir /home/afrumme1/CommonVoice_RIR/csv_output \
    --csv_filename utterance_mapping.csv \
    --room Silent_Room \
    --recording_filter Yamaha_Mixer
