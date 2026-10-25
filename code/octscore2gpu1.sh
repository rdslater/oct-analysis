#!/bin/bash
#SBATCH --job-name=octScore2
#SBATCH --partition=wrc
#SBATCH --account=wrc
#SBATCH --qos=wrc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=octScore2_%j.out
#SBATCH --error=octScore2_%j.err

set -e

# Prevent CPU thread thrashing in PyTorch/NumPy
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Activate Conda environment safely
source $(conda info --base)/etc/profile.d/conda.sh
conda activate /home/wisc/jiang452/.conda/envs/oct-score2

# Print GPU and environment diagnostic information
echo "Job started on node: $(hostname)"
echo "Using GPU: $CUDA_VISIBLE_DEVICES"

# Run Python script
python3 -u train.py --config configs/layers_3683.yaml 
