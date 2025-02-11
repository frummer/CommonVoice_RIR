#!/bin/bash

#SBATCH --job-name=compute-leakage-metric  # Job name
#SBATCH --nodes=1  # Number of nodes requested
#SBATCH --cpus-per-task=1   # Request 1 CPU core
#SBATCH --mem=8G            # Request 8GB of RAM
#SBATCH --partition=cpu  # Queue
#SBATCH --error=/home/afrumme1/CommonVoice_RIR/logs/logs_unsupervised_metrics/LDC_V1/leakage/error.log
#SBATCH --output=/home/afrumme1/CommonVoice_RIR/logs/logs_unsupervised_metrics/LDC_V1/leakage/output.log
#SBATCH --mail-user=afrumme1@jh.edu  # Email for reporting
#SBATCH --mail-type=END,FAIL

echo "Starting audio Separation Metric computation ..."
echo "Script started at: $(date)"

module purge 
module load conda 

# Ensure conda is properly initialized
conda activate /home/afrumme1/miniconda3/envs/common_voice_rir_3

# Define arguments with default values
AUDIO_DIR=/home/afrumme1/pipelines/separation_enhancement_pipeline/separation_LDC2014S02_V1_test_output
PREFIXES=""
SUFFIXES="_spk1_corrected _spk2_corrected"
TOP_X=3
METRIC="leakege"
WAV_ENDS_WITH="_corrected.wav"
WINDOW_SIZE=8000
WINDOW_STEP=4000
TOP_Z=10
OUTPUT_SUMMARY_DIR=/home/afrumme1/CommonVoice_RIR/output_dir/LDC_V1/leakage/summary
OUTPUT_DETAIL_DIR=/home/afrumme1/CommonVoice_RIR/output_dir/LDC_V1/leakage/detailed
SAVE_FULL_MATRIX=""

# Create output directories if they do not exist
mkdir -p "$OUTPUT_SUMMARY_DIR"
mkdir -p "$OUTPUT_DETAIL_DIR"

# Echo the arguments for verification
echo "Audio Directory: $AUDIO_DIR"
echo "Prefixes: $PREFIXES"
echo "Suffixes: $SUFFIXES"
echo "Top X: $TOP_X"
echo "Metric: $METRIC"
echo "WAV Ends With: $WAV_ENDS_WITH"
echo "Window Size: $WINDOW_SIZE"
echo "Window Step: $WINDOW_STEP"
echo "Top Z: $TOP_Z"
echo "Output Summary Directory: $OUTPUT_SUMMARY_DIR"
echo "Output Detail Directory: $OUTPUT_DETAIL_DIR"
echo "Save Full Matrix: $SAVE_FULL_MATRIX"

# Run the Python script with arguments
/home/afrumme1/miniconda3/envs/common_voice_rir_3/bin/python \
    /home/afrumme1/CommonVoice_RIR/src/evaluation_utils/unsupervised_metrics/leakage_metric.py \
    --audio_dir $AUDIO_DIR \
    --prefixes $PREFIXES \
    --suffixes $SUFFIXES \
    --top_x $TOP_X \
    --metric $METRIC \
    --wav_ends_with $WAV_ENDS_WITH \
    --window_size $WINDOW_SIZE \
    --window_step $WINDOW_STEP \
    --top_z $TOP_Z \
    --output_summary_dir $OUTPUT_SUMMARY_DIR \
    --output_detail_dir $OUTPUT_DETAIL_DIR \
    $SAVE_FULL_MATRIX

# Capture exit status
if [ $? -eq 0 ]; then
    echo "Audio metric computation completed successfully!"
else
    echo "Error: Audio metric computation failed." >&2
fi

echo "Script finished at: $(date)"
