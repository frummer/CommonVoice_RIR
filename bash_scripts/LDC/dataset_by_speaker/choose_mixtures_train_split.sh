#!/bin/bash

#SBATCH --job-name=choose-train-ldc-dataset  # Job name
#SBATCH --nodes=1  # Number of nodes requested
#SBATCH --cpus-per-task=1   # Request 1 CPU core
#SBATCH --mem=8G            # Request 8GB of RAM
#SBATCH --partition=cpu  # Queue
#SBATCH --error=/home/afrumme1/CommonVoice_RIR/logs/logs_ldc/choose_mixtures/train/error.log
#SBATCH --output=/home/afrumme1/CommonVoice_RIR/logs/logs_ldc/choose_mixtures/train/output.log
#SBATCH --mail-user=afrumme1@jh.edu  # Email for reporting
#SBATCH --mail-type=END,FAIL,BEGIN

echo "Started to choose train mixture for LDC ..."
echo "Script started at: $(date)"

module purge 
module load conda 

# Ensure conda is properly initialized
conda activate /home/afrumme1/miniconda3/envs/common_voice_rir_3

# Define arguments with default values
INPUT_CSV=/home/afrumme1/CommonVoice_RIR/output_dir/LDC_V2_dataset_creation/train_pairs.csv
CSV_OUT=/home/afrumme1/CommonVoice_RIR/output_dir/LDC_V2_dataset_creation/splits/train/train_dataset_mixtures.csv
SUMMARY_CSV=/home/afrumme1/CommonVoice_RIR/output_dir/LDC_V2_dataset_creation/splits/train/train_dataset_summary.csv
MIXTURES_AMOUNT=90000
RANDOM_STATE=42

# Echo the arguments for verification
echo "INPUT_CSV: $INPUT_CSV"
echo "CSV_OUT: $CSV_OUT"
echo "SUMMARY_CSV: $SUMMARY_CSV"
echo "MIXTURES_AMOUNT: $MIXTURES_AMOUNT"
echo "RANDOM_STATE: $RANDOM_STATE"

# Run the Python script with arguments
/home/afrumme1/miniconda3/envs/common_voice_rir_3/bin/python \
 /home/afrumme1/CommonVoice_RIR/src_ldc/v2/choose_mixtures_from_split.py \
    --csv_in $INPUT_CSV \
    --csv_out $CSV_OUT \
    --random_state $RANDOM_STATE \
    --summary_out $SUMMARY_CSV \
    --mixtures_amount $MIXTURES_AMOUNT  

# Capture exit status
if [ $? -eq 0 ]; then
    echo "Dataset split completed successfully!"
else
    echo "Error: Dataset split failed." >&2
fi

echo "Script finished at: $(date)"
