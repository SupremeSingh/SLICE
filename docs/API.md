# Shared critics and private training jobs

SLICE now provides a resident Python scorer, an immutable artifact registry, and
an optional private API for Slurm training. Operators can submit directly;
requests from API clients retain a separate approval step. The original
feature extraction, sampled-prefix training, Ridge equations and comparison
commands are unchanged. LSTD(lambda) is an opt-in readout over frozen features.

## 1. Install and test locally

From the `API` checkout:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,api,models,data,verify]'
python -m pytest -q
python -m slice_probe demo --out artifacts/api-demo
```

The tests instantiate a tiny random Qwen model locally; no pretrained weights,
datasets, Duke credentials, or GPU are needed. Optional integration dependencies
must be installed to run all tests. The synthetic demo is not a publishable model.
Use a new demo output directory when repeating it.

## 2. Publish an existing fitted probe

On the cluster, use a checkpoint produced by SLICE, its integrity sidecar, and
the matching feature manifest. For a completed matched comparison:

```bash
python -m slice_probe publish \
  --checkpoint artifacts/comparison/fitted/slice/sparse-seed42.pt \
  --calibration artifacts/comparison/fitted/calibration.json \
  --evaluation artifacts/comparison/evaluation \
  --feature-manifest artifacts/comparison/features/manifest.json \
  --registry /usr/xtmp/ms785/slice-critics --version math-v001

python -m slice_probe inspect \
  --artifact /usr/xtmp/ms785/slice-critics/math-v001
```

For `train` output, use its `sparse-seed42.pt` and omit `--calibration` unless you
have a matching calibrator. Use the actual feature path when comparison reused
an external cache. Historical metric exports are not checkpoints.

Publishing verifies checksums and atomically installs a complete directory.
Existing versions cannot be overwritten. The manifest records resolved blocks,
encoder asset hashes, prompt/tokenization, preprocessing, head conventions and
generation provenance. Back up the registry to durable storage; `/usr/xtmp` is
working storage. Published artifacts contain probe weights, not Qwen weights.
The question/split snapshot is included when supplied by the feature manifest.

## 3. Use it from another project

Install SLICE into the consuming project's Python environment:

```bash
python -m pip install '/usr/xtmp/ms785/SLICE[models]'
```

Inside that project's allocated GPU job:

```python
from slice_probe import Critic

critic = Critic.load(
    registry='/usr/xtmp/ms785/slice-critics',
    version='math-v001',
    device='cuda',
)
states = [dict(state_id='example', question='What is 12 times 8?', prefix='12 times 8 =')]
features = critic.encode_batch(states)  # float32 [N, width], no intercept column
scores = critic.score_batch(states)    # raw_value and explicit identity
# Only with a matching validation calibrator:
probabilities = critic.score_batch(states, return_calibrated_probability=True)
```

The encoder and compressor load once. Calls serialize hooks; states currently
run individually without padding to preserve existing extraction conventions.
Sharing artifact files does not share GPU memory between processes.
`encode_batch` raises on invalid input; `score_batch` reports input errors per
state. Neither truncates. Raw values are not probabilities.

Exact research inputs may instead supply the **entire** prompt-plus-prefix token
sequence, using `tokenizer_id=critic.tokenizer_id` and `token_ids=[...]`. Do not
also supply text. The tokenizer identity is returned by artifact inspection.

For many prefixes of one answer, use one causal encoder pass:

```python
# Exact IDs from the same tokenizer used by the artifact. No padding.
prompt_ids = [...]
response_ids = [...]
lengths = [len(prompt_ids) + t for t in range(len(response_ids))]
features = critic.encode_trajectory(
    prompt_ids + response_ids, lengths,
    tokenizer_id=critic.tokenizer_id, batch_size=1024,
)
```

Each length is the total number of visible tokens, including the prompt. The
first row sees the prompt only; the last sees all but the final response token.
Lengths may be unsorted or repeated, and output follows their input order.
Features remain float32 `[N, width]` without an intercept. Empty lengths return
zero rows. Invalid tokens, tokenizer identities, lengths and context overflow
raise errors. `batch_size` bounds compressor batches, not the encoder context.
Causal masking prevents future-token access; numerical roundoff can differ from
separate forwards. This interface supplies features, not transition rewards or
terminal conventions; consuming algorithms own those. Hooks remain serialized
within the resident instance. No rollout generation or policy training is added.

CLI equivalents:

```bash
python -m slice_probe score-batch \
  --artifact /usr/xtmp/ms785/slice-critics/math-v001 \
  --input examples/score-states.json --out artifacts/scores.json --device cuda
python -m slice_probe encode-batch \
  --artifact /usr/xtmp/ms785/slice-critics/math-v001 \
  --input examples/score-states.json --out artifacts/features.npy --device cuda
```

Pin the installed SLICE release/source revision for each experiment. Changing a
head does not change `feature_id`; changing the encoder, prompt, normalization,
resolved blocks or compressor does. Downstream training remains the consumer's
responsibility and does not modify registry artifacts.

## 4. Configure private job management

For this user's Duke CS cluster, start with the
[CS cluster profile and runbook](DUKE_CS.md). It includes the observed partition,
typed A5000 request and native Python launch; the generic example below is optional.

For direct Duke use, run `jobs --config artifacts/cluster-service.json init-cluster`
from the installed checkout on a login host, then use `submit` as shown below.
The [Duke guide](DUKE_CS.md#submit-directly-from-duke) covers node-local state and
reconnection. No SSH automation or running server is required.

Alternatively, run the controller on your computer. Install the same checkout and model,
dataset and verifier dependencies in the cluster environment too. No training
runs on a login node. Review your normal Slurm script to choose account,
partition, resources and any required environment/module setup.

```bash
mkdir -p artifacts
cp configs/service.example.json artifacts/service.json
```

Edit the settings:

- `database`: absolute path on your computer's local disk, not shared storage.
- `registry`: locally accessible registry for optional HTTP scoring. Training
  does not require this directory to contain the remote artifacts.
- `ssh_host`: your working SSH alias, e.g. `duke`, with authentication configured.
  Omit it only when running commands directly on the cluster.
- `cluster_root`: cluster job workspace; `cluster_registry`: published critics.
- `cluster_python`: absolute Python executable with this checkout installed.
- `account`, `partition`, `cpus`, `memory_gb`: your actual Slurm settings.
- `gres`: one GPU, either `gpu:1` or a typed request such as `gpu:a5000:1`.
- `cluster_source`: optional absolute `src` directory. When set, approval copies
  `slice_probe` into the job workspace and workers import that snapshot.
- `container_runtime`, `container_image`, `container_binds`: optional Apptainer
  or Singularity execution. Omit runtime/image together for native execution.
- `max_total_minutes`: conservative cumulative allowance across submitted jobs.

Each submitted attempt reserves its full wall-time allowance, including failed
and cancelled work. This is not measured billing. One job may be active globally;
uncertain submissions also hold that slot. Raise the allowance deliberately in
the settings when needed. Do not keep SQLite on an NFS/Lustre filesystem.

Managed requests default to a bounded pilot service: **10–1,000 questions**, 2–8 answers
per question, at most 3,584 new tokens, up to three training seeds, 30 epochs,
width 768, and 1–1,440 requested minutes. There is one active job **per ledger**,
including uncertain submissions, with a default cumulative allowance of 1,440
reserved minutes. These limits do not describe the earlier large studies or
restrict unrelated Slurm jobs. Research requests support up to **13,000 total
questions**, with explicit `train_questions`, `validation_questions` and
`test_questions` summing to `max_questions`, and an operator `max_questions`
setting raised above the default 1,000. The full Numina preset is
`configs/training.numina-10k.json`: 10,000/1,000/2,000 questions. The research
service example proposes a 43,200-minute cumulative reservation allowance and
96 GB RAM. Keep the existing database, host and history when updating limits;
do not initialize a new controller or overwrite working settings with a template.

The 24-hour per-request limit remains. These configuration changes do not add
batched generation, generation arrays, partial feature extraction, or training
checkpoints. A fresh full-size serial run may exceed that limit. Reuse original
saved data or implement bounded resumable stages before submitting the full
preset. Confirm the cluster's partition time limits first.

Recovery reuses only completed stages with matching checksums. Collection can
resume from complete per-question answer batches; an interrupted question's batch
is regenerated. Other interrupted stages restart from their beginning. There
are no within-epoch compressor or partial Ridge-matrix checkpoints. The original
deadline persists across worker requeues, including elapsed time between attempts.
Requeue therefore does not reset the time budget. Repeating `submit` returns the
old attempt; it is not a retry command. See the [Duke recovery procedure](DUKE_CS.md#controller-backup-and-recovery).

## 5. Submit a training request

No API or running server is needed for a first cluster trial:

```bash
python -m slice_probe jobs --config artifacts/service.json create \
  --request configs/training.example.json --project my-project \
  --idempotency-key first-pilot
python -m slice_probe jobs --config artifacts/service.json show --id JOB_ID
python -m slice_probe jobs --config artifacts/service.json approve --id JOB_ID
python -m slice_probe jobs --config artifacts/service.json refresh --id JOB_ID
```

For a trusted operator, combine creation and approval in one explicit command:

```bash
python -m slice_probe jobs --config artifacts/cluster-service.json submit \
  --request configs/training.example.json --project my-project \
  --idempotency-key first-pilot
```

`submit` authorizes compute immediately. Repeating the same project, key and
request returns the existing attempt without another submission, including when
the submission outcome is unknown. Changed requests with the same key fail.
Resource bounds, the single active-job slot, source snapshots and cumulative
budget reservations apply to both submission paths. API clients still create
requests awaiting operator approval.

`show` previews resolved blocks and maximum workload. `approve` is the explicit
operator action that allocates compute; creating a request never submits work.
Initial dataset/split validation occurs on the compute worker after approval;
the operator should review the dataset and columns before submitting.

The example is a new 100-question pilot with 70/15/15 splits. Numina delegates
directly to the original `prepare-numina` implementation: verified, nonsynthetic
`cn_k12` word problems, matching single boxed references, family deduplication,
the full eligible-family scan and `slice-v1` ordering. Other datasets retain the
generic scalar-reference selector with a 100,000-row scan limit. Revisions are
pinned; invalid/insufficient datasets fail explicitly.

Training reuses `fit_comparison`: all matched baselines, a supervised sparse
compressor, validation-only choices, and the existing pre-test lock. The selected
Ridge head is refitted with the original `all_token_ridge` after the compressor
is frozen, as in `train --answers`. The sampled comparison checkpoints remain
immutable and separately labeled. The selected head receives validation-only
calibration before its test evaluation. Both
its results and comparison baselines are reported. The preset runs all matched
arms, so its compute cost exceeds fitting a single head.
Collection defaults match `collect`: eight responses, 3,584 new tokens and seed
271828. Compressor fitting defaults to three seeds (42/43/44), with the first
seed's sparse compressor published as the resident critic; comparison metrics
still report every seed. The example allows twelve hours. `generation_seed`,
`training_seeds` and `ridge_states` are explicit request fields; choosing
`ridge_states: sampled` requests the separately labeled sampled-prefix ablation.
An allocation-local smoke gate runs before dataset processing. It records the
actual model, layer selection, source, packages and optional container checksum;
training refuses a mismatched gate even without scheduler dependencies.

`layer_base` controls geometric spacing; `num_slices` selects the count. The
resolved block list is authoritative. Explicit invalid selections fail. Width,
epochs, response count, token count and wall time are bounded in requests.
New model/layer/compression choices train new artifacts.

The registry version is `job-JOB_ID`. Retrieve its path from job status, then
load it directly from your other cluster project. Downloads are optional.

## 6. Optional HTTP API

Keep credentials outside settings and source control. Generate a secret with
`python -c 'import secrets; print(secrets.token_urlsafe(32))'`, then set:

```bash
export SLICE_SERVICE_CONFIG="$PWD/artifacts/service.json"
export SLICE_API_KEYS='{"my-project":"REPLACE_WITH_A_LONG_RANDOM_SECRET"}'
uvicorn slice_probe.api:create_app --factory --host 127.0.0.1 --port 8000
```

Use one server worker. The service binds locally; remote projects need an SSH
tunnel/private connection. Do not expose an unauthenticated public port. Operator
approval is intentionally CLI-only. Interactive docs are at `/docs`.

```python
import os
from slice_probe.client import Client

client = Client('http://127.0.0.1:8000', os.environ['MY_SLICE_KEY'])
job = client.create_training_job(
    dict(dataset='AI-MO/NuminaMath-1.5', max_questions=100,
         layer_base=2, num_slices=5),
    idempotency_key='experiment-001',
)
print(client.job(job['id']))
```

Reuse the same idempotency key and payload after a timeout; changed payloads
with that key are rejected. Each project can access only its own managed jobs
and artifacts. Manually published artifacts without an owner are shared among
your project keys. The optional scoring endpoint uses the **local** registry and
local device; it does not silently reserve a cluster GPU.

Endpoints: `POST /v1/training-jobs`, `GET /v1/jobs/{id}`, cancellation and
artifact paths under that job, `GET /v1/usage`, `GET /v1/critics/{version}`, and
`POST /v1/scoring-batches`. Training and cancellation require `Idempotency-Key`.
Scoring is read-only and has no persistent side effects.

## 7. Cancellation and recovery

```bash
python -m slice_probe jobs --config artifacts/service.json cancel --id JOB_ID
```

Cancellation uses `scancel`, ending current work and preventing later stages.
Slurm enforces requested wall time. Jobs use `--requeue`; when Slurm requeues an
allocation, the worker verifies completed stage hashes and resumes. Partial
collection resumes per question. Other partial stages restart from scratch;
partial accumulated matrices are never appended twice. Each job has a worker
lock so overlapping attempts cannot update the same workspace.
An additional persisted elapsed-time deadline spans requeues, including time
waiting between attempts, so restarts cannot reset the job's allowance.

`submission_failed` means failure before any `sbatch` attempt.
`submission_unknown` means the scheduler may have accepted the job. Run
`refresh` to reconcile its unique name against `squeue` and `sacct`; never launch
a duplicate blindly. Scheduler/accounting outages retain the uncertain state.
If no job ever appears, investigate manually; absence is not automatic proof of
failure. The control plane never retries an uncertain submission automatically.

Back up both the local database and published registry. A local API restart
does not stop a submitted Slurm job. The cluster Python environment, installed
source and settings must remain consistent across requeues.

## 8. Optional LSTD(lambda)

Set `"head": "lstd"` and `"trace_lambda": 0.9` in a managed training request.
Managed correctness jobs require gamma=1. The compressor remains supervised and
frozen; only the final head is replaced. Ridge remains the default.

For explicit extraction and fitting:

```bash
python -m slice_probe extract-trajectories \
  --checkpoint artifacts/probes/sparse-seed42.pt --answers artifacts/answers \
  --feature-manifest artifacts/features/manifest.json \
  --out artifacts/trajectories --device cuda
python -m slice_probe fit-lstd \
  --checkpoint artifacts/probes/sparse-seed42.pt \
  --trajectories artifacts/trajectories \
  --feature-manifest artifacts/features/manifest.json \
  --trace-lambda 0.9 --discount-gamma 1 --regularization 0.01 \
  --out artifacts/lstd
```

Extraction uses consecutive pre-action token states of training answers, not
the sampled-prefix cache. It requires the exact representation-training
collection, rejects modified answers, and excludes capped or unknown-termination
answers. Rewards are zero except for verified correctness on the final transition.
Terminal next features, including the intercept, are zero; traces reset per answer.
LSTD therefore has a different fitting distribution from the sampled-prefix
baselines. Reports label that distinction; they are not a matched-weighting
comparison of Ridge versus LSTD. Both are evaluated on the same held-out states.

Statistics use float64 raw sums:

```text
e_t = gamma * lambda * e_(t-1) + phi_t
A += e_t (phi_t - gamma * phi_next)^T
b += e_t reward_t
(A + epsilon I) w = b
```

The solver handles a nonsymmetric matrix and reports residuals, conditioning and
transition counts. Lambda/gamma changes require new statistics; cached trajectory
features can be reused. The low-level fitter allows discounted targets, which
must not be interpreted as undiscounted correctness probabilities. No policy
training, changing actor or arbitrary matrix-upload API is introduced.

Managed LSTD publication includes its statistics and trajectory manifest. When
publishing manually, pass `--statistics artifacts/lstd/statistics.npz` and
`--trajectory-manifest artifacts/trajectories/manifest.json` alongside the new
checkpoint. A calibrator for the old Ridge checkpoint cannot be reused.
