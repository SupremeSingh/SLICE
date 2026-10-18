# Reproducing and testing SLICE

## What is and is not released

The historical table is a metric export, **not yet an exact reproducibility release**.
Its test questions were inspected during development. The standalone repository
currently does not include the original question-ID file, exact dataset/model
commit SHAs, fitted checkpoints or individual predictions from that run. Do not
infer those missing values from today's upstream model or dataset. See
[`results/artifact-status.json`](../results/artifact-status.json).

The workflow below creates a new, auditable comparison. It deliberately changes
the historical fitting protocol: **every method trains on the same sampled states
with the same row weights**. New scores must not silently replace the historical
table or be described as exact reproduction. Generation still costs time; none of
these commands manufacture the missing historical data.

## One command for the complete comparison

After collecting answers as described in the README:

```bash
python -m slice_probe compare --answers artifacts/answers \
  --config configs/matched-prefix.json --out artifacts/comparison --device auto
```

This shares extraction across all representations, fits all methods,
locks validation choices, evaluates, and writes `evaluation/report.txt` and
`evaluation/metrics.csv`. Optional `--figures` writes reliability/risk-coverage
plots after installing `.[figures]`. For a software walkthrough with the bundled
sample, use `configs/smoke.json` instead. Sample results are not scientific evidence.

Methods: final-layer Ridge; two-thirds-layer Ridge; normalized concatenation
Ridge; a ten-linear-map residual classifier; plain bottleneck + Ridge; sparse
bottleneck + Ridge; and a constant training-prior baseline. Ridge baselines are
deterministic; neural heads use the configured seeds. Head metrics are averaged,
not their predictions. The residual network has one input map, four two-map
residual blocks and one output map, with GELU nonlinearities.

The model is frozen. The final-layer baseline includes its learned final RMSNorm;
the two-thirds baseline reads the raw intermediate output at block `ceil(2D/3)`.
Concatenation uses per-block RMS normalization without the learned final gain.
Feature conventions, actual blocks and dimensions are recorded in the manifest.
The compressor and residual probe share epoch, optimizer, learning-rate candidate,
seed and state budgets; each selects checkpoints using validation Brier. Ridge
uses the same fixed raw-sum epsilon across representations. Normalization uses
training data only. A shared budget is not a claim of optimal tuning for every arm.

To reuse a cache instead of extracting again:

```bash
python -m slice_probe extract --answers artifacts/answers \
  --baselines --checkpoints 0 32 128 --out artifacts/matched-features
python -m slice_probe compare --features artifacts/matched-features \
  --config configs/matched-prefix.json --out artifacts/comparison-cached
```

Older caches without baseline arrays cannot support all comparisons. Re-extract
from the saved answers; do not substitute different sampled prefixes.

## What does a partial answer add beyond question difficulty?

`summary.json` provides two complementary controls:

1. **Carry forward the question-only score.** Each question's L=0 score is reused
   at every later prefix. Its Brier error is paired with the updated prefix score
   on exactly the same rows/outcomes. A negative `updated_minus_initial` difference
   favors information from the partial answer. No labels construct this baseline.
2. **Rank continuations of the same question at the same token count.** Extra
   evaluation-only states at L=0,32,128 never enter training. These are recomputed
   from truncated inputs, caching identical prefixes within each question, to avoid
   ranking floating-point variations from different future sequence lengths. Within each question
   containing both correct and incorrect continuations at that length, AUROC
   measures their separation; question AUROCs are averaged equally. The initial
   score ties within a question and has AUROC 0.5. Sustained improvement above
   this control would support information beyond question difficulty.

All heads see the same diagnostic states. Prefixes must precede termination;
results at L=128 condition on reaching L=128. Report mixed-question counts and
intervals. Small or empty groups are not evidence of success. These diagnostics
are observational, not causal evidence or a guarantee of generalization. Standard
`evaluate` also reports carry-forward scores, but exact-position coverage from
randomly sampled prefixes alone can be sparse; use `compare` for fixed checkpoints.

The number-template split filter does not establish semantic or pretraining
decontamination. Test freshness requires a genuinely uninspected question split,
not merely different sampled answers for previously inspected questions.

## Matched fitting versus the historical all-token ablation

`compare` always fits **every method on sampled prefixes**, without extra token
weights for long answers. The older `train --answers ...` command remains an
explicit, separately labeled **all-training-token Ridge readout ablation** for
plain/sparse compressors. It is not part of the matched comparison and does not
make the residual head's fitting distribution identical. Keep its scores in a
separate report. Every selected checkpoint records its fitting distribution.

## Calibration and useful decisions

Raw Ridge scores are retained unchanged. With `calibration: "platt"`, a monotone
logistic calibrator is fitted separately to each selected head's validation
predictions. Test data never fits the calibrator. The release contains both raw
scores and calibrated probabilities, reliability bins, test Brier/AUROC/ECE and
risk-coverage curves. Only calibrated output should be described as a predicted
probability. For single-prefix inference, the matching calibrator is optional:

```bash
python -m slice_probe predict --checkpoint artifacts/comparison/fitted/slice/sparse-seed42.pt \
  --calibration artifacts/comparison/fitted/calibration.json \
  --question 'There are 9 avocados, with 3 per serving. How many servings?'
```

The calibrator must identify the checkpoint by hash. Without it, the command
returns only a raw score/classification, not a confidence percentage. Calibration
on validation does not guarantee calibration after a
change of dataset or generation policy.

Risk means the fraction of accepted answers that are wrong; acceptance prefers
high predicted correctness. Ties enter together. A validation-selected empirical
10% risk operating point is applied unchanged to test outcomes. Its test coverage
and risk are reported; this is **not** a finite-sample risk guarantee. States and
answers share questions, so they are not independent observations.

## Lock a confirmation run before evaluation

Copy the configuration, set `mode` to `confirmation`, and describe the new question
split in `test_scope`. Freeze this choice before examining test performance:

```bash
python -m slice_probe comparison-fit --features artifacts/matched-features \
  --config configs/confirmation.json --out artifacts/locked-fit
# Save the printed LOCK SHA256 outside the output directory.
python -m slice_probe comparison-evaluate --features artifacts/matched-features \
  --run artifacts/locked-fit --lock-sha256 SAVED_SHA256 \
  --out artifacts/confirmation-results
```

Fitting reads train/validation arrays only. The lock hashes the resolved
configuration, selected checkpoints, calibrators, feature manifest and code/runtime
inventory before test evaluation. Confirmation requires the externally saved hash
and unchanged code. A one-command `compare` also creates its lock before evaluation,
but separate commands make the boundary easier to review. Hashes do not prove a
user has never inspected a test set or establish statistical superiority.

## Release artifacts for review without repeating generation

```bash
python -m slice_probe release-bundle --answers artifacts/answers \
  --features artifacts/comparison/features --run artifacts/comparison/fitted \
  --evaluation artifacts/comparison/evaluation --out artifacts/release
python -m slice_probe verify-bundle --bundle artifacts/release \
  --out artifacts/release-check
```

The bundle contains exact question IDs/splits/texts, generation settings, pinned
model/dataset identity when collected via `prepare-numina`, selected checkpoints,
validation calibration, configuration/environment/code hashes, and per-prefix
raw/calibrated predictions with labels, question IDs, response IDs and positions.
`verify-bundle` checks the pre-test lock and question/collection identities,
then recomputes the comparison metrics and question-difficulty/within-question
controls from those predictions without model inference. Integrity checks do not establish correct verifier labels.

Feature arrays and base-model weights are excluded by default. Add
`--include-features` to allow re-evaluation directly from cached activations
without extraction; this can add tens of GB. Add `--include-answers` to
include saved answer texts/token IDs for label review, subject to data licensing.
Release the bundle as a versioned artifact and cite its URL/hash; the command
itself does not upload anything. Pin an immutable source-code version alongside
it. For cached-input comparisons, pass that feature directory instead. Bundles
from new runs do not retroactively publish the missing historical artifacts.

## Tests

```bash
python -m pip install -e '.[dev,models,verify]'
python -m pytest -q
```

Core tests use controlled data to check split isolation, lock tampering, metric
recomputation, carry-forward controls and within-question ranking. Optional
integration tests instantiate a real three-block Qwen2 transformer locally (random
weights; no model downloads) and exercise actual feature extraction/causality.
Real Math-Verify tests cover correct/incorrect, missing/malformed and ambiguous
final answers. Timeout boundary tests inject a controlled timeout rather than
relying on a machine-dependent slow expression. Optional tests skip when their
libraries are absent; the separate integration CI job installs them.

The collector now records `single-scalar-box-v3`: malformed boxes, conflicting
boxed answers, and explicit numeric final-answer alternatives are excluded and
audited. A plain final scalar contradicting an earlier numeric box is also excluded. This conservative policy can exclude
otherwise usable answers. It differs from historical labels and must be reported
with exclusions. It does not certify intermediate mathematical reasoning.
