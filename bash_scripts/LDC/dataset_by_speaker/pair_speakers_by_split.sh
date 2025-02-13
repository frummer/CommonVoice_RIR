#!/bin/bash

#SBATCH --job-name=split-ldc-dataset  # Job name
#SBATCH --nodes=1  # Number of nodes requested
#SBATCH --cpus-per-task=1   # Request 1 CPU core
#SBATCH --mem=8G            # Request 8GB of RAM
#SBATCH --partition=cpu  # Queue
#SBATCH --error=/home/afrumme1/CommonVoice_RIR/logs/logs_ldc/split_dataset_by_speaker_v2/error.log
#SBATCH --output=/home/afrumme1/CommonVoice_RIR/logs/logs_ldc/split_dataset_by_speaker_v2/output.log
#SBATCH --mail-user=afrumme1@jh.edu  # Email for reporting
#SBATCH --mail-type=END,FAIL

echo "Starting LDC2014S02 dataset split ..."
echo "Script started at: $(date)"

module purge 
module load conda 

# Ensure conda is properly initialized
conda activate /home/afrumme1/miniconda3/envs/common_voice_rir_3

# Define arguments with default values
INPUT_CSV=/home/afrumme1/CommonVoice_RIR/output_dir/LDC_V2_dataset_creation/ldc_splits.csv
OUTPUT_DIR=/home/afrumme1/CommonVoice_RIR/output_dir/LDC_V2_dataset_creation


# Echo the arguments for verification
echo "Base Directory: $BASE_DIR"
echo "Output CSV: $OUTPUT_CSV"

# Run the Python script with arguments
/home/afrumme1/miniconda3/envs/common_voice_rir_3/bin/python \
 /home/afrumme1/CommonVoice_RIR/src_ldc/v2/pair_speakers_by_split.py \
    --input_csv $INPUT_CSV \
    --output_dir $OUTPUT_DIR \

# Capture exit status
if [ $? -eq 0 ]; then
    echo "Dataset split completed successfully!"
else
    echo "Error: Dataset split failed." >&2
fi

echo "Script finished at: $(date)"
