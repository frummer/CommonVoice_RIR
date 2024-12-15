#!/bin/bash 

# 

#SBATCH --job-name=separation #job name
#SBATCH --nodes=1  #number of nodes requested
#SBATCH --gpus=1  #number of gpus requested
#SBATCH --partition=gpu-a100   #queue
#SBATCH --error=logs/extract.err
#SBATCH --output=logs/extract.out
#SBATCH --mail-user=afrumme1@jh.edu  #email for reporting
#SBATCH --account=a100acct  #account


echo "Starting mms inference script"
module purge 
module load conda 

# Set CUDA_HOME environment variable
#export CUDA_HOME=/usr/local/cuda  # adjust if CUDA is installed in a different path
#export PATH=$CUDA_HOME/bin:$PATH
#export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PAT
export CUDA_VISIBLE_DEVICES=0
source activate /home/afrumme1/miniconda3/envs/mms

/home/afrumme1/miniconda3/envs/mms/bin/python /home/afrumme1/CommonVoice_RIR/src/evaluate_sep_enh_pipeline.py