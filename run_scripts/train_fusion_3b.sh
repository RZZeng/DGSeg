#!/bin/bash
#SBATCH --job-name=train_fus_3b
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --output=runs/train_fusion_3b_%j.log

source "${CONDA_SH:-${HOME}/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-dgseg}"

cd "$( cd "$( dirname "${BASH_SOURCE[0]}" )/.." && pwd )"

echo "Python: $(which python)"
echo "CUDA devices: $(python -c 'import torch; print(torch.cuda.device_count())')"

DGSEG_ROOT="${DGSEG_ROOT:-$PWD}" python src/train_fusion/train_sam3_fusion_qwen25_3b.py

echo "Fusion training completed with exit code: $?"
