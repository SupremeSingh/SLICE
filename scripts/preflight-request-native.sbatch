#!/usr/bin/env bash
# CPU-only data/tokenizer check; no GPU allocation or model inference.
#SBATCH --job-name=slice-request-preflight
#SBATCH --partition=compsci
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from SLICE}"
: "${SLURM_JOB_ID:?Run through sbatch}"
export PYTHONUNBUFFERED=1
export SLICE_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$SLICE_NUM_THREADS"
job_dir="${1:?Pass the previous managed job directory}"
srun "$PWD/.venv-native/bin/python" -m slice_probe preflight-request \
  --job-dir "$job_dir" --out "$PWD/artifacts/request-preflight-${SLURM_JOB_ID}"
