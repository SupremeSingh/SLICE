#!/usr/bin/env bash
# Submit from SLICE: sbatch scripts/parity-native.sbatch /absolute/completed/job
#SBATCH --job-name=slice-native-parity
#SBATCH --partition=compsci-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the SLICE checkout}"
: "${SLURM_JOB_ID:?Run through sbatch}"
job_dir="${1:?Pass the absolute completed managed job directory}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export SLICE_NUM_THREADS="$OMP_NUM_THREADS"
hostname
srun "$PWD/.venv-native/bin/python" -m slice_probe parity \
  --job-dir "$job_dir" --out "$PWD/artifacts/parity-${SLURM_JOB_ID}" --device cuda:0
