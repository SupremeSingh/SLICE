#!/usr/bin/env bash
# Submit from SLICE after mkdir -p logs. No model downloads or training.
#SBATCH --job-name=slice-native-gpu-check
#SBATCH --partition=compsci-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script from the SLICE checkout}"
: "${SLURM_JOB_ID:?Run through sbatch}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

hostname
"$PWD/.venv-native/bin/python" - <<'PY'
import os

import torch

print('PyTorch:', torch.__version__)
print('CUDA build:', torch.version.cuda)
print('CUDA_VISIBLE_DEVICES:', os.environ.get('CUDA_VISIBLE_DEVICES'))
assert torch.cuda.is_available(), 'CUDA is unavailable in this GPU allocation'
assert torch.cuda.device_count() == 1, 'Expected exactly one Slurm-visible GPU'
print('GPU:', torch.cuda.get_device_name(0))
x = torch.arange(8, device='cuda:0', dtype=torch.float32)
result = x.square().sum().item()
assert result == 140.0, f'Unexpected GPU calculation: {result}'
print('GPU calculation:', result)
print('Native GPU validation passed.')
PY
