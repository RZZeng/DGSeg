#!/bin/bash
#SBATCH --job-name=train_fus_7b
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=runs/train_fusion_7b_%j.log

source "${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-dgseg}"

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

echo "=== Job Info ==="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Python: $(which python)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
echo "Start time: $(date)"

DGSEG_ROOT="${DGSEG_ROOT:-$PWD}" python src/train_fusion/train_sam3_fusion_qwen25_7b.py

echo "Fusion training completed with exit code: $?"
echo "End time: $(date)"
