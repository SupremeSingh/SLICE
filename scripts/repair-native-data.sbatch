#!/usr/bin/env bash
# Repair only SLICE's Arrow dependency after its training job has stopped.
#SBATCH --job-name=slice-native-data-repair
#SBATCH --partition=compsci
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the SLICE checkout}"
: "${SLURM_JOB_ID:?Run through sbatch}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
python="$PWD/.venv-native/bin/python"
mkdir -p artifacts
"$python" -m pip freeze > "artifacts/data-repair-${SLURM_JOB_ID}-before.txt"
# Match the data extra; preserve the existing NumPy, Torch and other packages.
"$python" -m pip install --index-url https://pypi.org/simple \
  --only-binary=:all: --no-deps 'pyarrow==23.0.1'
"$python" -m pip check
"$python" -m pytest -q \
  tests/test_integrations.py::test_dataset_runtime_numpy_arrow_compatibility \
  tests/test_integrations.py::test_data_runtime_failure_precedes_model_loading
"$python" -c 'import numpy, pyarrow, datasets; print("NumPy:", numpy.__version__, "Arrow:", pyarrow.__version__, "datasets:", datasets.__version__)'
"$python" -m pip freeze > "artifacts/data-repair-${SLURM_JOB_ID}-after.txt"
echo 'Native dataset runtime validation passed.'
