# Duke CS cluster setup

This profile uses the user's observed **CS cluster** settings: login hosts
`compsci-login-01/02`, CPU partition `compsci`, GPU partition `compsci-gpu`,
and one A5000 requested as `--gres=gpu:a5000:1`. These are not live availability
checks or defaults for other Duke clusters.

## Use the existing original SLICE

The original 768-wide study is
`/usr/xtmp/ms785/rl_ms_project/outputs/numina-sparse-768-v1`, using source data
under `outputs/jlens-large-numina-v2` in that project. To serve those trained
features and scores, import its existing compressor and all-token Ridge weights
instead of launching another training request:

```bash
cd /usr/xtmp/ms785/SLICE
mkdir -p logs
sbatch scripts/import-original-native.sbatch
```

This CPU job checks the original protocol, selected checkpoints, fitted-head
checksums, model/tokenizer assets, question/prompt snapshot and saved test cache.
It validates converted predictions on 512 evenly spaced original test rows per
seed before publishing any heads. It performs no generation, fitting, relabeling
or modification of original files. Validation details are in
`artifacts/original-import-JOB_ID/validation.json`; the log lists published paths.
The upstream model revision is not guessed: local original assets remain pinned
by their saved SHA256 hashes. Existing registry versions are never overwritten.

All three original seeds are published as `original-sparse768-seed42`,
`original-sparse768-seed43` and `original-sparse768-seed44`. Load the desired
artifact with `Critic.load(registry='/usr/xtmp/ms785/slice-critics',
version='original-sparse768-seed42', device='cuda:0')` in a consuming compute job.
Raw Ridge values remain raw; no new probability calibration is invented.
The historical three-head average is separate from any single published head.

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

## Submit directly from Duke

Log in normally with `ssh ms785@login.cs.duke.edu`, then connect to the controller
host if the portal placed you elsewhere:

```bash
ssh ms785@compsci-login-01
hostname
```

Run initialization on **compsci-login-01**, using the installed native environment:

```bash
cd /usr/xtmp/ms785/SLICE
source .venv-native/bin/activate
python -m slice_probe jobs --config artifacts/cluster-service.json init-cluster --host compsci-login-01
python -m slice_probe jobs --config artifacts/cluster-service.json submit \
  --request configs/training.example.json --project slice-pilot \
  --idempotency-key native-pilot-001
```

`init-cluster` checks the requested hostname before creating anything and records
the current Python executable and source checkout. It creates a private SQLite ledger in `/var/tmp/slice-controller-UID`
on that host, with jobs and critics under sibling `slice-service` and
`slice-critics` directories on working storage. `/var/tmp` must be node-local;
use `--state-dir` to select another node-local directory if needed. Existing
configuration is reused. There is no SSH hop or persistent server.

`submit` explicitly authorizes compute and submits one GPU job. The example
requests 100 questions selected by the original `prepare-numina` recipe, eight
responses each and at most 3,584 generated tokens per response. It uses generation
seed 271828, training seeds 42/43/44, base 2, five layers, width 768 and an
all-training-token Ridge head. It reserves one A5000, 64 GB RAM and up to twelve
hours. The smaller 70/15/15 question counts bound this pilot; selection filters
and ordering are the original implementation. Edit the request to choose another
dataset, budget, or head. Repeating the same key and request returns the existing
attempt, including failed and uncertain attempts; it never launches a retry.
Changed requests require a new key. Concurrent requests still respect the single
active-job slot and cumulative allowance.

Save the returned `id` (SLICE request) and `scheduler_id` (Slurm job). To monitor:

```bash
python -m slice_probe jobs --config artifacts/cluster-service.json refresh --id JOB_ID
python -m slice_probe jobs --config artifacts/cluster-service.json list
```

You may disconnect while training runs. Reconnect to the controller hostname
printed during initialization for subsequent controller commands. Configuration
rejects another host or a missing ledger rather than silently resetting job and
budget history. Node-local storage is not a durable backup; use the recovery
procedure below. Shared request/status files and published artifacts
remain in the job workspace and registry.

From another project, use the absolute configuration and request paths. For
example, the same command works in `/usr/xtmp/ms785/rl_ms_project` with
`--config /usr/xtmp/ms785/SLICE/artifacts/cluster-service.json` and an absolute
`--request` path. Install SLICE in that project's environment before calling it.

## Check managed versus standalone training

Reuse a completed Ridge job's saved questions and answers. This audit performs
no generation and changes no original job or registry artifacts. From a login
prompt in the checkout, submit a separate bounded GPU audit:

```bash
mkdir -p logs
sbatch scripts/parity-native.sbatch /usr/xtmp/ms785/slice-service/jobs/fdb57872fc80423c81bdacd3a5a2be10
```

The audit requests one A5000, 64 GB and six hours. It is a direct Slurm job,
separate from the controller's training-request allowance. Outputs are
`artifacts/parity-SLURM_JOB_ID/report.json` and
`logs/slice-native-parity-SLURM_JOB_ID.out`. It independently extracts features
and calls standalone `training.fit` with the recorded seeds, hyperparameters and
readout distribution. It compares array values and labels, normalization,
compressor weights and validation-selected trial names for every seed, selected
Ridge weights, and held-out raw predictions. Stored managed predictions are
checked too. Checkpoint file bytes differ because timing and provenance differ.

`matches: true` requires numerical agreement and matching core source, package,
Python, thread and CUDA identities. Tolerances default to `rtol=1e-5, atol=1e-6`
and are recorded alongside absolute errors and bitwise equality. A mismatch
leaves the report and returns nonzero; an interrupted audit leaves partial
outputs and needs a fresh output directory. It does not automatically resume.
This checks the wrapper against the current standalone implementation on the
same saved data, not the historical experiment or predictive usefulness.

## Controller backup and recovery

The on-cluster CLI automatically snapshots the local ledger before submission
and after successful create/submit/approve/refresh/cancel commands, under
`artifacts/controller-backups/`. Snapshots use SQLite's consistent backup API;
they include configuration identity and a checksum, not copied live WAL files.
Preserve configuration and snapshots on durable storage outside `/usr/xtmp`.
For an explicit named snapshot, run on **compsci-login-01**:

```bash
python -m slice_probe jobs --config artifacts/cluster-service.json backup \
  --out artifacts/controller-backups/before-parity
```

If the local ledger is lost, stop new submissions. Keep the existing shared
configuration, inspect `squeue`, `sacct` and shared job directories, and choose
the latest valid snapshot. On the same controller host:

```bash
python -m slice_probe jobs --config artifacts/cluster-service.json restore \
  --backup artifacts/controller-backups/before-parity
```

Restore never overwrites an existing ledger and rejects checksum/configuration
mismatches, a different host, or a snapshot missing a staged shared request.
For each request ID returned in `reconcile`, run `jobs ... refresh --id ID`
before submitting new work. Known staged nonterminal requests are held as
`submission_unknown`, preserving reservations and duplicate protection. Missing
scheduler records do not release that hold automatically. If restore refuses a
stale snapshot, recover a newer ledger and reconcile with Slurm; do not delete
the configuration, create a new ledger, or retry with a new key to bypass it.

Managed requests default to 1,000 questions and allow one active job per ledger.
Research requests can opt into 13,000 total questions with explicit split sizes
and an increased controller `max_questions` limit. Only collection
resumes within a stage, from complete per-question files. Feature extraction,
compressor fitting, all-token Ridge, evaluation and publishing restart an
unfinished stage; completed stages are reused only after checksum checks.
Requeues retain the original deadline, and failed/cancelled attempts retain
their full requested GPU-minute reservation. Use standalone commands for larger
studies; resource changes require deliberate planning.

## Full-size training and layer selection

`layer_base` and `num_slices` already configure layers counted from the end.
For 28 blocks, base 2 with five slices resolves to blocks `[28,27,25,21,13]`;
base 3 with three slices resolves to `[28,26,20]`. Base 1 uses consecutive final
blocks by convention. Invalid combinations fail rather than duplicate blocks.
Changing the resolved blocks requires a new feature cache and compressor.

The full training preset `configs/training.numina-10k.json` explicitly selects
10,000 training, 1,000 validation and 2,000 test questions, eight responses each,
and the original 30 epochs and three seeds. The operator example
`configs/service.duke-cs.research.example.json` raises total question capacity to
13,000, RAM to 96 GB and the cumulative allowance to 43,200 reserved GPU-minutes
(30 one-GPU days). This is a proposed upper bound, not a reservation or runtime
estimate. Preserve the existing ledger and host when updating settings.

Do not submit this full preset to the current serial collector merely by
increasing its allowance: each request still has a 24-hour ceiling and requeues
retain their deadline. The original study used batched generation and sharded
statistics. Prefer the saved original 10k-question data; fresh large generation
needs a bounded stage/shard execution plan. Check partition limits first:

```bash
sinfo -p compsci-gpu -o "%P %l %G"
```

## Optional controller on your computer

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

## Optional separate review and approval

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

## Repair an incompatible dataset runtime

With NumPy 1.26.4, use the data extra's pinned PyArrow 23.0.1. A newer Arrow
build may import-fail despite `pip check` passing. The managed smoke gate now
checks actual NumPy/Arrow/datasets interoperability before loading Qwen.

Once all jobs using `.venv-native` have stopped, submit this CPU-only repair:

```bash
mkdir -p logs
sbatch scripts/repair-native-data.sbatch
```

It replaces only PyArrow, records before/after packages, and runs the dataset
runtime checks. Require `COMPLETED`, `0:0` and the validation-passed log message
before submitting a new training request with a new idempotency key. Preserve
the old failed request and outputs. No GPU check or NumPy/Torch upgrade is needed.

## Optional existing container support

`configs/service.duke-cs.example.json` retains the older Apptainer/Singularity
profile for experiments that explicitly need it. Set the image, runtime, matching
Python environment and binds there; that mode also records the image checksum.
It is not needed for the native setup above.
