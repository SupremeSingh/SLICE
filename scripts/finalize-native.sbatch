#!/usr/bin/env bash
# Separate, bounded GPU follow-up; submit only after the original job leaves squeue.
#SBATCH --job-name=slice-finalize
#SBATCH --partition=compsci-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a5000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from SLICE}"
finalization="${1:?Pass the absolute directory returned by prepare-finalization}"
[[ "$finalization" = /* ]] || { echo 'Finalization directory must be absolute'; exit 1; }
export PYTHONUNBUFFERED=1
export SLICE_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OMP_NUM_THREADS="$SLICE_NUM_THREADS"
srun env PYTHONPATH="$finalization/source" "$PWD/.venv-native/bin/python" \
  -m slice_probe finalize-run --plan "$finalization/plan.json" --device cuda
