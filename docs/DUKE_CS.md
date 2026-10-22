# Duke CS cluster setup

This profile uses the user's observed **CS cluster** settings: login hosts
`compsci-login-01/02`, CPU partition `compsci`, GPU partition `compsci-gpu`,
and one A5000 requested as `--gres=gpu:a5000:1`. These are not live availability
checks or defaults for other Duke clusters.

## Check the cluster

After logging in through your working SSH connection:

```bash
hostname
pwd
sinfo -p compsci-gpu -o "%P %a %l %D %G"
module avail
df -h /usr/xtmp/ms785
```

Use login nodes only for editing, light inspection and submission. Install or
build substantial dependencies inside a compute allocation. The preferred setup
is now a dedicated native environment at `SLICE/.venv-native`; it does not use a
container. Existing jobs and their environments stay unchanged. SLICE requests
only one GPU for training.

Before choosing native Python/PyTorch packages, inspect a CPU compute node:

```bash
cd /usr/xtmp/ms785/SLICE
mkdir -p logs
sbatch scripts/inspect-native.sbatch
```

Read `logs/slice-native-inspect-JOB_ID.out` after completion. The script only
reports the native Python/packages and loaded modules; it installs nothing and
does not touch other projects. CUDA availability must later be verified in a GPU
allocation. If submission times out, reconcile the job named
`slice-native-inspect` in `squeue`/`sacct` before retrying.

## Validate the native environment

After installing dependencies, submit CPU tests and a separate brief GPU check
from the checkout containing the installed environment:

```bash
cd /usr/xtmp/ms785/SLICE
mkdir -p logs
sbatch scripts/test-native.sbatch
sbatch scripts/check-gpu-native.sbatch
```

The CPU job writes each test name and memory usage to
`logs/slice-native-tests-JOB_ID.out`. It copies only package sources for the
standalone test, excludes runtime environments and caches, and keeps temporary
test files on working storage. The GPU job checks PyTorch and a small CUDA
calculation in `logs/slice-native-gpu-check-JOB_ID.out`; it downloads no models.
Neither job changes the environment or other running jobs.

Check both returned IDs with `sacct` and read their logs. Require `COMPLETED`
and `0:0` for both jobs before proceeding. If submission times out, reconcile
the corresponding job name before retrying. A host-memory OOM or time limit
does not establish a CUDA or SLICE training failure.

## Configure the controller on your computer

```bash
mkdir -p artifacts
cp configs/service.duke-cs.native.example.json artifacts/service.json
```

Fill in:

- Local `database` and `registry` paths.
- Your existing SSH alias in `ssh_host`; the example's `duke-cs` is a placeholder.
- The native environment's Python executable in `cluster_python`. The example
  uses `/usr/xtmp/ms785/SLICE/.venv-native/bin/python`.
- `account` only if your normal CS jobs require an explicit account. Otherwise,
  leave it null to use your configured Slurm default.

The observed `/usr/project/xtmp/ms785` path may refer to the same storage as
`/usr/xtmp/ms785`; use consistent paths in the native environment and configuration.

Install this SLICE revision and optional model/data/verifier dependencies in that
environment once, from a compute allocation. The [API guide](API.md) covers the
library and installation commands.

## Review before submission

```bash
python -m slice_probe jobs --config artifacts/service.json create \
  --request configs/training.example.json --project my-project \
  --idempotency-key cs-pilot-001
python -m slice_probe jobs --config artifacts/service.json show --id JOB_ID
```

`show` includes the exact resources, batch script and container command, without
connecting to the cluster or submitting anything. Check the paths and workload.

```bash
python -m slice_probe jobs --config artifacts/service.json approve --id JOB_ID
python -m slice_probe jobs --config artifacts/service.json refresh --id JOB_ID
```

Approval copies the small `src/slice_probe` package into the job directory,
creates the log directory before `sbatch`, and writes a strict Bash script.
The native script uses `srun`, the explicit environment Python path and thread
counts matching the requested CPUs. No dependency
installation or model inference takes place on the login node.

Before dataset processing, a compute-node smoke check verifies the actual GPU,
model context, resolved layers, causal features and Math-Verify. Its result is
tied to the request, source hashes and packages. Later stages
refuse a mismatched gate. Collection uses the exact smoke-tested model revision.
The smoke report is preserved in the published critic metadata.

## Monitor and recover

Run these after logging into the cluster:

```bash
squeue -j SLURM_JOB_ID
sacct -j SLURM_JOB_ID --format=JobID,JobName%40,State,ExitCode,Elapsed
tail -n 100 /usr/xtmp/ms785/slice-service/jobs/JOB_ID/slurm-SLURM_JOB_ID.log
```

`JOB_ID` is SLICE's request ID; `SLURM_JOB_ID` is the numeric scheduler ID.
Completion requires `COMPLETED`, exit code `0:0`, and a worker completion record.
Missing exit accounting remains `completion_unknown`. Missing application status
can indicate a node/allocation launch failure; inspect the Slurm log before retrying.

A submission timeout remains `submission_unknown`, even if SSH returned an error.
Reconcile the unique name `slice-JOB_ID` with `squeue` and `sacct`. Never blindly
resubmit. Cancel a specific request with the controller's `jobs ... cancel`
command; do not use a blanket user-wide cancellation.

Completed stage hashes support safe requeue recovery. Preserve failed outputs and
wait for cancellation cleanup before changing environments. Source snapshots,
published critics and reports should be backed up separately from `/usr/xtmp`.

## Optional existing container support

`configs/service.duke-cs.example.json` retains the older Apptainer/Singularity
profile for experiments that explicitly need it. Set the image, runtime, matching
Python environment and binds there; that mode also records the image checksum.
It is not needed for the native setup above.
