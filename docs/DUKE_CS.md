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
build substantial dependencies inside a compute allocation. Use the Python
environment created for your pinned container, and never modify a shared
environment while jobs use it. SLICE itself requests only one GPU.

## Configure the controller on your computer

```bash
mkdir -p artifacts
cp configs/service.duke-cs.example.json artifacts/service.json
```

Fill in:

- Local `database` and `registry` paths.
- Your existing SSH alias in `ssh_host`; the example's `duke-cs` is a placeholder.
- The pinned `.sif` path and its matching environment's Python executable.
- `container_runtime`: `apptainer` or `singularity`, whichever is installed.
- Binds covering the job workspace, source, image, environment, caches and registry.
  The image must also be readable inside the container for checksum verification.
- `account` only if your normal CS jobs require an explicit account. Otherwise,
  leave it null to use your configured Slurm default.

The observed `/usr/project/xtmp/ms785` path may refer to the same storage as
`/usr/xtmp/ms785`; confirm path accessibility inside the container. Do not change
the scientific configuration to work around a missing bind.

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
The script uses `srun`, the configured container's `--nv`, explicit binds and
Python path, and thread counts matching the requested CPUs. No dependency
installation or model inference takes place on the login node.

Before dataset processing, a compute-node smoke check verifies the actual GPU,
model context, resolved layers, causal features and Math-Verify. Its result is
tied to the request, source hashes, packages and container checksum. Later stages
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
