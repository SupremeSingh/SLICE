#!/usr/bin/env bash
# Submit from SLICE after mkdir -p logs. This suite uses CPU only.
#SBATCH --job-name=slice-native-tests
#SBATCH --partition=compsci
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script from the SLICE checkout}"
: "${SLURM_JOB_ID:?Run through sbatch}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export SLICE_NUM_THREADS="$OMP_NUM_THREADS"
export CUDA_VISIBLE_DEVICES=''
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1

python="$PWD/.venv-native/bin/python"
[[ -x "$python" ]] || { echo 'Missing .venv-native/bin/python'; exit 1; }
mkdir -p artifacts
temporary=$(mktemp -d "$PWD/artifacts/pytest-native-${SLURM_JOB_ID}.XXXXXX")

hostname
"$python" -m pip check
# Require the source-only checkout copy before placing test output in artifacts.
"$python" -m pytest -q \
  tests/test_workflow.py::test_standalone_copy_does_not_traverse_environments_or_test_output

runner=("$python" -m pytest -vv -x --durations=10 --basetemp "$temporary")
if [[ -x /usr/bin/time ]]; then
  exec /usr/bin/time -v "${runner[@]}"
fi
exec "${runner[@]}"
