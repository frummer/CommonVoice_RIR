#!/bin/bash

#SBATCH --job-name=compute-audio-metrics  # Job name
#SBATCH --nodes=1  # Number of nodes requested
#SBATCH --cpus-per-task=1   # Request 1 CPU core
#SBATCH --mem=8G            # Request 8GB of RAM
#SBATCH --partition=cpu  # Queue
#SBATCH --error=/home/afrumme1/CommonVoice_RIR/logs/logs_unsupervised_metrics_ar_cv/unseparation_logs/error.log
#SBATCH --output=/home/afrumme1/CommonVoice_RIR/logs/logs_unsupervised_metrics_ar_cv/unseparation_logs/output.log
#SBATCH --mail-user=afrumme1@jh.edu  # Email for reporting
#SBATCH --mail-type=END,FAIL

echo "Starting audio Separation Metric computation  ..."
# Echo the current time
echo "Script started at: $(date)"

module purge 
module load conda 

# Ensure conda is properly initialized
conda activate /home/afrumme1/miniconda3/envs/common_voice_rir_3

# Define arguments
AUDIO_DIR=/home/afrumme1/pipelines/separation_enhancement_pipeline/separation_output_prev
PREFIXES=""
SUFFIXES="_spk1_corrected _spk2_corrected"
TOP_X=3
METRIC="unseparation"
WAV_ENDS_WITH="_corrected.wav"
OUTPUT_DIR=/home/afrumme1/CommonVoice_RIR/output_dir
# Echo the arguments for verification
echo "Audio Directory: $AUDIO_DIR"
echo "Prefixes: $PREFIXES"
echo "Suffixes: $SUFFIXES"
echo "Top X: $TOP_X"
echo "Metric: $METRIC"
echo "WAV Ends With: $WAV_ENDS_WITH"
echo "Output Directory: $OUTPUT_DIR"

# Run the Python script with arguments
/home/afrumme1/miniconda3/envs/common_voice_rir_3/bin/python \
    /home/afrumme1/CommonVoice_RIR/src/evaluation_utils/unsupervised_metrics/unseparation_metric.py \
    --audio_dir $AUDIO_DIR \
    --prefixes $PREFIXES \
    --suffixes $SUFFIXES \
    --top_x $TOP_X \
    --metric $METRIC \
    --wav_ends_with $WAV_ENDS_WITH \
    --output_dir $OUTPUT_DIR

# Capture exit status
if [ $? -eq 0 ]; then
    echo "Audio metric computation completed successfully!"
else
    echo "Error: Audio metric computation failed." >&2
fi

echo "Script finished at: $(date)"
