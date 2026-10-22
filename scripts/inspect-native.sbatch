#!/usr/bin/env bash
# Submit from the SLICE checkout after creating logs/. No installation or training.
#SBATCH --job-name=slice-native-inspect
#SBATCH --partition=compsci
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit this script from the SLICE checkout}"
: "${SLURM_JOB_ID:?Run this inspection through sbatch}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

hostname
pwd
command -v python3
python3 --version

python3 - <<'PY'
import importlib.util
from importlib.metadata import PackageNotFoundError, requires, version
import os
import sys

print('Python:', sys.version)
print('Executable:', sys.executable)
print('Loaded modules:', os.environ.get('LOADEDMODULES', '(none)'))
print('ensurepip available:', importlib.util.find_spec('ensurepip') is not None)
for package in ('pip', 'torch', 'numpy', 'sympy'):
    try:
        print(f'{package}: {version(package)}')
    except PackageNotFoundError:
        print(f'{package}: not installed')

if importlib.util.find_spec('torch') is not None:
    import torch
    print('PyTorch path:', torch.__file__)
    print('PyTorch compiled CUDA version:', torch.version.cuda)
    print('PyTorch SymPy requirements:', [r for r in requires('torch') or []
                                         if r.lower().startswith('sympy')])

print('CPU environment inspection only; no GPU availability check was performed.')
PY
