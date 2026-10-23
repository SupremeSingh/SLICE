# Next Duke run

These steps use the native environment and existing controller ledger. Run them
from `/usr/xtmp/ms785/SLICE` on **compsci-login-01**. If login lands on 02, reconnect
to 01 before controller commands. No interactive allocation is needed: `sbatch`
requests the compute allocations for each script.

## 1. Pull and submit the bounded checks

```bash
cd /usr/xtmp/ms785/SLICE
git pull --rebase origin API
source .venv-native/bin/activate
mkdir -p logs
sbatch scripts/test-native.sbatch
sbatch scripts/extraction-parity-native.sbatch
sbatch scripts/preflight-request-native.sbatch \
  /usr/xtmp/ms785/slice-service/jobs/78a51f3cbe214d4794dc087292854c22
```

Record the three returned scheduler IDs. The CPU tests exercise both Ridge and
LSTD publication and restart behavior. The GPU extraction audit checks the
original saved compressors against fresh Qwen extraction. The CPU request
preflight pins the failed run's model/dataset revisions and selects the full
10,000/1,000/2,000 questions while excluding prompts that exceed the existing
512-token/context budget. It never truncates prompts or reduces the training count.

Check each job using its returned numeric ID:

```bash
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed
```

Require all three to complete with `0:0`. Inspect:

- `logs/slice-native-tests-TEST_JOB_ID.out`: tests passed.
- `artifacts/extraction-parity-PARITY_JOB_ID/report.json`: `matches: true`.
- `artifacts/request-preflight-PREFLIGHT_JOB_ID/report.json`: `ready: true` and counts 10,000 train, 1,000 validation, 2,000 test.

The parity report records observed numerical errors and original code/model
hashes. Passing it supports extraction agreement for the tested tokens and
precision; it does not claim historical dataset/generation/verifier equivalence.

## 2. Use original pretrained SLICE, if desired

After extraction parity passes, publish the existing original heads once:

```bash
sbatch scripts/import-original-native.sbatch
```

This CPU job does no training. Require success and a passing saved-prediction
check. It publishes `original-sparse768-seed42`, `original-sparse768-seed43`, and
`original-sparse768-seed44` in `/usr/xtmp/ms785/slice-critics`. If these versions
already exist, use them without rerunning publication. A consumer can then load:

```python
from slice_probe import Critic

critic = Critic.load(
    '/usr/xtmp/ms785/slice-critics/original-sparse768-seed42', device='cuda',
)
features = critic.encode_trajectory(
    token_ids, prefix_lengths, tokenizer_id=critic.tokenizer_id,
)
```

The consumer runs inference inside its own GPU allocation. It does not need the
controller or an online server to read a published artifact.

## 3. Retry the full training request after checks pass

Set `PREFLIGHT_JOB_ID` to the numeric ID returned for the CPU preflight above.
Settle the failed job's reservation against final Slurm elapsed-time accounting;
this preserves its request and failure history rather than reserving another
28 days against the unchanged 30-day cumulative service allowance.

```bash
python -m slice_probe jobs --config artifacts/cluster-service.json settle \
  --id 78a51f3cbe214d4794dc087292854c22
python -m slice_probe jobs --config artifacts/cluster-service.json submit \
  --request "artifacts/request-preflight-${PREFLIGHT_JOB_ID}/request.json" \
  --project slice-research --idempotency-key numina-10k-002
```

Use the generated pinned request, not the old example file. The new request
retains 10,000 training questions, eight responses, 3,584 generated tokens,
768-wide compression, 30 epochs and the three training seeds. Record its new
SLICE ID and Slurm ID. Refresh using that new SLICE ID on controller host 01.
Do not resubmit if the scheduler outcome is unknown; reconcile first. Repeating
this exact idempotency key returns the existing request instead of duplicating it.

The original published critic remains usable while this independent study runs.
Neither importing nor retraining modifies original study outputs or other jobs.
