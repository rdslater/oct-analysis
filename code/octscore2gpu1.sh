#!/bin/bash
#SBATCH --mem-per-cpu=16G
#SBATCH --cpus-per-task=20
#SBATCH --gpus-per-node=1
#SBATCH --job-name=octScore2      # Name of the job
#SBATCH --output=octScore2_output_%j.log   # Where to save the output
#SBATCH --error=octScore2_error_%j.log     # Where to save errors
#SBATCH --nodes=1                   # Run on a single node
#SBATCH --ntasks=1                  # Run a single task
#SBATCH --time=06:05:00             # Time limit (5 minutes)

eval "$(conda shell.bash hook)"
conda activate .score2
python3 train.py
