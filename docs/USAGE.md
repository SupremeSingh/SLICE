# Protocol and artifact guide

For matched runnable baselines, question-difficulty controls, calibration and locked
confirmation, see [the reproducibility guide](REPRODUCIBILITY.md).

## Installation

Install from this folder with Python 3.10+:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[models,data,verify,dev]'
```

Install a PyTorch wheel suitable for your hardware before this command when your
system needs a particular CUDA build. `models` pins the Transformers/tokenizer
versions used by the original feature pipeline. The core demo requires only
NumPy and PyTorch. Tests do not download models or datasets.

`python -m slice_probe --help` lists every command. Output directories must be empty
except `collect`, which can reuse completed per-question answer batches with the
same settings. Partial extraction/fitting should use a new output directory.

## Questions and labels

Question JSONL records require `id`, `question`, numeric `reference`, and
`split` (`train`, `val`, or `test`). All three splits must be present. IDs and
number-template families must be unique across the entire file.

`prepare-numina` selects `cn_k12`, nonsynthetic, valid math word problems with a
single numeric boxed reference matching the saved answer. It rejects obvious
nontext/proof and multiple-answer prompts. These are conservative heuristics,
not a proof of correct dataset labels. The dataset commit is recorded; supply
`--revision COMMIT_SHA` to reuse an exact selection source. This selector does
not recreate earlier exclusions against other datasets or the exact published
study splits. No answer-success filtering is used for question selection.

`collect` downloads and pins a model snapshot, records asset hashes, uses the
completion prompt in `features.py`, and samples complete answers at temperature 1.
The reference implementation uses Hugging Face generation serially, not a large
inference service. Eight answers per question and up to 3,584 new tokens are
expensive; use `--responses` and `--max-new-tokens` to bound a pilot. Prompts must
fit 512 tokens and prompt-plus-answer must fit the model context.

Math-Verify provides final-answer labels. Parse/verification failures and empty
answers are recorded and excluded from probe fitting/evaluation, not silently
labeled incorrect. Inspect statuses and raw saved answers. These labels do not
certify individual reasoning steps.

## Frozen causal features

`extract` loads the same model in FP32 eager mode, with TF32 disabled, and checks
asset hashes. One hidden vector is read at the last token of each available
prefix. Hooks capture post-block states before final model normalization. All
selected block vectors are RMS-normalized before concatenation.

Positions are L=0 once, followed by seven iid uniform draws from 1..T-1 with
replacement (all zero for a one-token answer). The seed is a stable function of
question index and answer index. Sampling is reward-blind and excludes terminal
states. Future answer length controls offline sampling only; future tokens are
never inputs to an earlier state. Tests compare full-sequence and prefix-only
features.

Caches contain NumPy arrays per split: `x`, `y`, `question`, `response`, `position`.
The manifest records layer order, model identity, counts and SHA256 hashes.
Each retained answer contributes eight rows; repeated rows are possible by design.
A 13,000-question cache can exceed 25 GB. Loading uses memory mapping; integrity
checks read complete files and can take time on shared storage.

## Fitting and evaluation

`train` uses training-only mean and a single isotropic scale. Each plain/sparse
seed gets the same 30-epoch budget and learning-rate candidates. The temporary
classifier's validation Brier selects an epoch. Validation Brier after a
sampled-prefix Ridge fit selects a learning rate per representation/seed.
The trained model never sees test outcomes during fitting.

By default, the final Ridge uses sampled training states. Supplying `--answers`
recomputes every pre-action training-token state after locking the compressor;
compressed features update raw `A = sum(phi phi^T)` and `b = sum(phi y)` with
`phi = [z, 1]`. Solve `(A + 0.01 I) w = b` in float64; do not invert the matrix.
This includes intercept regularization and no division by state count. Each
prefix's label is its complete answer's outcome. Neither method changes the LLM.

`evaluate` reads selected checkpoints only, verifies data identities, and checks
question separation across splits. It writes per-prefix predictions, per-head
and mean accuracy/Brier/AUROC/ECE, prefix-position groups, and paired question
bootstrap comparisons. Head averaging means averaging their metrics, not
ensembling their predictions. Correlated answers and prefixes stay together in
the bootstrap. Intervals are descriptive and not multiplicity-adjusted.

## Move a fitted probe

Training produces self-contained `.pt` checkpoints and `.sha256.json` integrity
sidecars. Copy both files together, or copy the whole probe directory to retain
its validation selection manifest. Prediction uses the saved block selection,
compression weights, normalization and Ridge weights. The base model is downloaded
by its saved identifier/revision or supplied with `predict --model /path/to/model`;
its asset hashes must match. Base-model weights are not embedded in the probe.
No second repository or historical experiment directory is required.

## Devices and small examples

All model, fitting and prediction commands accept `--device auto`, `cpu`,
`cuda:0` or `mps`. Auto prefers CUDA, then Apple Silicon, then CPU. Feature capture
uses FP32. CUDA answer generation uses BF16 when supported and FP32 otherwise.
Ridge solves run in CPU float64; MPS readout arithmetic also runs on CPU because
MPS does not support float64. CPU is the most portable fallback. Device-specific
kernels can still produce numerical differences; test before a large study.

`demo` uses synthetic features without downloads. `sample` packages seven authored
questions and fourteen answers with explicit human-written 0/1 labels, and pins a
small model for tokenization and subsequent extraction. It needs `[models]`, but
not `[data]` or `[verify]`. These illustrative labels are not generated-model
outcomes. The source is bundled inside the installed package; the command works
outside a repository checkout. See the README for complete example commands.

`extract --layer-base B --num-slices K` selects depths independently of the
compressor width. Use `layers --num-layers D` with the same options to preview.
Distances from the final block follow `max(previous + 1, floor(B ** i))`, with
`previous=0` initially and `B >= 1`. Out-of-range distances stop selection. An
explicit K must be achievable; it is never silently padded or clipped. The actual
layer list and selection rule are stored in the feature manifest and checkpoint.
Supported encoders have a Qwen-style decoder `layers` structure and RMS-normalized
hidden features; arbitrary transformer architectures are not automatically supported.

## Research results versus a fresh run

`results/reported-results.json` contains the completed experiment's reported
metrics, not a fitted checkpoint or predictions for new inputs. The CPU demo
is synthetic; the bundled math sample is authored and intentionally tiny. A fresh standalone run changes question selection and uses a
different generation implementation, so its scores should not be expected to
match the table exactly. Keep the generated answers, feature manifests and fitted checkpoints together
for reproducibility of your own study.
