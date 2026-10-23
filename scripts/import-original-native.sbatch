#!/usr/bin/env bash
# Package existing original weights; no model generation, fitting or GPU use.
#SBATCH --job-name=slice-import-original
#SBATCH --partition=compsci
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the SLICE checkout}"
: "${SLURM_JOB_ID:?Run through sbatch}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export SLICE_NUM_THREADS="$OMP_NUM_THREADS"
srun "$PWD/.venv-native/bin/python" -m slice_probe import-original \
  --study /usr/xtmp/ms785/rl_ms_project/outputs/numina-sparse-768-v1 \
  --model /usr/xtmp/ms785/rl_ms_project/models/qwen-math \
  --registry /usr/xtmp/ms785/slice-critics \
  --out "$PWD/artifacts/original-import-${SLURM_JOB_ID}"
