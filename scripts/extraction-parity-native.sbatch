#!/usr/bin/env bash
#SBATCH --job-name=slice-extraction-parity
#SBATCH --partition=compsci-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from SLICE}"
export PYTHONUNBUFFERED=1
export SLICE_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$SLICE_NUM_THREADS"
srun "$PWD/.venv-native/bin/python" -m slice_probe extraction-parity \
  --study /usr/xtmp/ms785/rl_ms_project/outputs/numina-sparse-768-v1 \
  --model /usr/xtmp/ms785/rl_ms_project/models/qwen-math \
  --original-root /usr/xtmp/ms785/rl_ms_project \
  --out "$PWD/artifacts/extraction-parity-${SLURM_JOB_ID:?}" --device cuda
