#!/bin/bash 

# 

#SBATCH --job-name=create-test-en-commonvoice-separation-dataset #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --partition=cpu  #queue
#SBATCH --cpus-per-task=4   # Request 4 CPU cores
#SBATCH --mem=32G           # Request 32GB of RAM
#SBATCH --error=/home/afrumme1/CommonVoice_RIR/logs/logs_ldc/logs_ldc_dataset/test/extract.err
#SBATCH --output=/home/afrumme1/CommonVoice_RIR/logs/logs_ldc/logs_ldc_dataset/test/extract.out
#SBATCH --mail-user=afrumme1@jh.edu  #email for reporting
#SBATCH --mail-type=END,FAIL,BEGIN

echo "Creating LDC 2-spk Mixtures dataset"
echo "Script started at: $(date)"

module purge 
module load conda 

conda activate /home/afrumme1/miniconda3/envs/common_voice_rir_3
export PYTHONPATH=/home/afrumme1/CommonVoice_RIR:$PYTHONPATH
CONFIG_PATH=/home/afrumme1/CommonVoice_RIR/src_ldc/configs/create_ldc_overlapped_test_set_config.json
echo "CONFIG_PATH: $CONFIG_PATH"

/home/afrumme1/miniconda3/envs/common_voice_rir_3/bin/python \
    /home/afrumme1/CommonVoice_RIR/src_ldc/create_overlapped_dataset.py  \
    --config_path $CONFIG_PATH


# Capture exit status
if [ $? -eq 0 ]; then
    echo "Creating LDC 2-spk Mixtures dataset completed successfully!"
else
    echo "Error: Creating LDC 2-spk Mixtures dataset failed." >&2
fi

echo "Script finished at: $(date)"
