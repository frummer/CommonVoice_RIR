#!/bin/bash

#SBATCH --job-name=compute-ellipsis-metric  # Job name
#SBATCH --nodes=1  # Number of nodes requested
#SBATCH --cpus-per-task=1   # Request 1 CPU core
#SBATCH --mem=8G            # Request 8GB of RAM
#SBATCH --partition=cpu  # Queue
#SBATCH --error=/home/afrumme1/CommonVoice_RIR/logs/logs_unsupervised_metrics/LDC_V1/ellipsis/error.log
#SBATCH --output=/home/afrumme1/CommonVoice_RIR/logs/logs_unsupervised_metrics/LDC_V1/ellipsis/output.log
#SBATCH --mail-user=afrumme1@jh.edu  # Email for reporting
#SBATCH --mail-type=END,FAIL

echo "Starting Ellipsis Metric computation ..."
echo "Script started at: $(date)"

module purge 
module load conda 

# Ensure conda is properly initialized
conda activate /home/afrumme1/miniconda3/envs/common_voice_rir_3

# Define arguments with default values
SEPARATED_AUDIO_DIR=/home/afrumme1/pipelines/separation_enhancement_pipeline/separation_LDC2014S02_V1_test_output
MIXTURES_AUDIO_DIR=/export/fs05/afrumme1/babylon_datasets_1/LDC2014S02/V1_split/test/mixture
PREFIXES=""
SUFFIXES="_spk1_corrected _spk2_corrected"
WAV_ENDS_WITH="_corrected.wav"
TOP_X=3
OUTPUT_DIR=/home/afrumme1/CommonVoice_RIR/output_dir/LDC_V1/ellipsis

# Create output directory if it does not exist
mkdir -p "$OUTPUT_DIR"

# Echo the arguments for verification
echo "Separated Audio Directory: $SEPARATED_AUDIO_DIR"
echo "Mixtures Audio Directory: $MIXTURES_AUDIO_DIR"
echo "Prefixes: $PREFIXES"
echo "Suffixes: $SUFFIXES"
echo "WAV Ends With: $WAV_ENDS_WITH"
echo "Top X: $TOP_X"
echo "Output Directory: $OUTPUT_DIR"

# Run the Python script with arguments
/home/afrumme1/miniconda3/envs/common_voice_rir_3/bin/python \
    /home/afrumme1/CommonVoice_RIR/src/evaluation_utils/unsupervised_metrics/ellipsis_metric.py \
    --separated_audio_dir $SEPARATED_AUDIO_DIR \
    --mixtures_audio_dir $MIXTURES_AUDIO_DIR \
    --prefixes $PREFIXES \
    --suffixes $SUFFIXES \
    --wav_ends_with $WAV_ENDS_WITH \
    --top_x $TOP_X \
    --output_dir $OUTPUT_DIR

# Capture exit status
if [ $? -eq 0 ]; then
    echo "Ellipsis metric computation completed successfully!"
else
    echo "Error: Ellipsis metric computation failed." >&2
fi

echo "Script finished at: $(date)"
