# SLICE

**Sparse Latent Inference for Correctness Estimation**

Can a small reader of a language model's internal states predict whether its
unfinished answer will be correct?

SLICE combines hidden states from several depths of a **frozen language model**,
compresses them into **768 learned features by default**, and fits a **linear Ridge readout**
to predict eventual final-answer correctness. The question alone can also be
scored, before any answer tokens are generated.

```text
Question + optional partial answer
                  ↓
         Frozen language model
                  ↓
    Five normalized hidden-layer states
                  ↓
       7,680 → 768 learned features
                  ↓
       Linear correctness readout
```

![SLICE overview](docs/figures/overview.png)

The language model never changes. Completed answers supply training labels;
only the question and available prefix are needed at prediction time.

## What we found

On 2,000 held-out NuminaMath question families, the reported frozen-model study
produced the following **correctness-prediction** scores:

| Representation + readout | Features | Accuracy ↑ | AUROC ↑ | Brier ↓ |
|---|---:|---:|---:|---:|
| Constant training-prior baseline | — | 55.48% | 0.500 | 0.24705 |
| Final layer + Ridge | 1,536 | 66.89% | 0.737 | 0.21306 |
| Two-thirds layer + Ridge | 1,536 | 70.27% | 0.773 | 0.19702 |
| Five normalized layers + Ridge | 7,680 | 72.01% | 0.790 | 0.18917 |
| Plain bottleneck + Ridge | 768 | 72.63% | 0.800 | 0.18529 |
| **SLICE: sparse bottleneck + Ridge** | **768** | **72.66%** | **0.801** | **0.18505** |
| Five normalized layers + ten-layer residual probe | 7,680 | 73.30% | 0.804 | 0.17918 |

**Before generating an answer**, SLICE reached **70.49% accuracy / 0.766 AUROC**,
versus **67.09% / 0.721** for two-thirds-layer Ridge. These predict outcomes across
sampled continuations; they do not mean the language model answered 70.49% correctly.

The study generated 104,000 answers across 10,000 training, 1,000 validation and
2,000 test questions. It retained 102,329 verifiable answers and evaluated 125,920
test prefixes. Bottleneck scores average three feature-training seeds sharing the
same data. The test set was inspected during development; results condition on
verifiable answers. Plain and sparse accuracy is nearly identical. The deeper
residual probe has better Brier score and calibration.

Compression reduces feature width **10×**. The final readout has **769 parameters**,
but the compressor has about **5.9 million**. Sparse activations do not imply faster
execution, and these results do not establish an end-to-end inference speedup.
[Detailed results and limitations](results/README.md) · [Machine-readable evidence](results/reported-results.json)

## Start in a few commands

Python **3.10+** is required. No cluster, scheduler or account on a particular
institution's machines is needed. Clone this repository and enter its directory:

```bash
git clone https://github.com/SupremeSingh/SLICE.git
cd SLICE
```

Create an environment and run a CPU-only check:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
python -m pytest -q
python -m slice_probe demo --out artifacts/demo
```

On Windows, activate with `.venv\Scripts\Activate.ps1` in PowerShell instead of
`source .venv/bin/activate`. Use `python` if `python3` is unavailable.

The demo runs on CPU with synthetic features and no downloads. It checks the
complete compressor → Ridge → evaluation workflow. **Its scores are not research
results.** Both `python -m slice_probe` and the installed `slice-probe` command work.

To work with real model activations, additionally install:

```bash
python -m pip install -e '.[models,data,verify]'
```

Commands accept `--device auto` (the default), `cpu`, `cuda:0`, or `mps`.
Automatic selection prefers an NVIDIA GPU, then Apple Silicon, then CPU. Install
a [PyTorch build matching your hardware](https://pytorch.org/get-started/locally/)
if needed. A GPU is recommended for larger model runs; CPU works for small examples. Model and dataset downloads require network access
on their first use.
They remain subject to their own licenses. Large caches and fitted weights are
not bundled in this repository.

## Try the included sample data

The repository includes **seven authored math questions and fourteen labeled
answers**, covering correct and incorrect outcomes in all three splits. No dataset
account or verifier is needed. This example downloads the small Qwen2.5-0.5B model
on first use and extracts real activations:

```bash
python -m pip install -e '.[models]'
python -m slice_probe sample --out artifacts/sample-answers
python -m slice_probe extract --answers artifacts/sample-answers --num-slices 3 \
  --out artifacts/sample-features
python -m slice_probe train --features artifacts/sample-features --width 32 --seeds 42 \
  --learning-rates 0.001 --epochs 5 --batch-size 32 --out artifacts/sample-probes
python -m slice_probe evaluate --features artifacts/sample-features \
  --probes artifacts/sample-probes --out artifacts/sample-evaluation
```

Read the [sample answers](src/slice_probe/sample_answers.jsonl) or the tokenized
`artifacts/sample-answers/answers/*.json` files. These tiny, hand-written examples
verify that the software works; **their scores do not measure the base model's
own correctness or establish generalization**. Use the downloadable Numina workflow
below for an actual study. The no-download `demo` above remains the fastest check.

## Choose the slices

Set **`--layer-base`** to control the geometric spread and **`--num-slices`** to
control how many blocks to read. Preview the selection without downloading a model:

```bash
python -m slice_probe layers --num-layers 28 --layer-base 2 --num-slices 5
# {"layers": [28, 27, 25, 21, 13]}
python -m slice_probe layers --num-layers 28 --layer-base 3 --num-slices 3
# {"layers": [28, 26, 20]}
```

Pass the same options to `extract`. Blocks are one-based, ordered from the end;
the final block is always included. Distances start at 1 and grow as
`floor(base ** i)`, advancing by at least one to avoid duplicates. Base 1 selects
consecutive final blocks. If `--num-slices` is omitted, all in-range distances are
used. An explicit count that cannot fit raises an error. Defaults preserve the
five-block base-2 selection for a 28-block model.

Input width becomes `number of slices × model hidden size`; `train --width` sets
the compressed width independently. Both are saved with the checkpoint, so
prediction automatically uses the fitted selection. More slices increase feature
storage and compressor cost. The results table and diagram describe the original
five-block, 768-feature configuration.

## Train your own probe

The workflow is **select questions → collect answers → extract prefixes → train
compressor → fit Ridge → evaluate**. For a small first run:

```bash
python -m slice_probe prepare-numina --train 200 --val 50 --test 100 --out data/questions
python -m slice_probe collect --questions data/questions/questions.jsonl --responses 4 \
  --model Qwen/Qwen2.5-Math-1.5B --device auto --out artifacts/answers
python -m slice_probe extract --answers artifacts/answers --device auto --out artifacts/features
python -m slice_probe train --features artifacts/features --width 768 --seeds 42 \
  --answers artifacts/answers --device auto --out artifacts/probes
python -m slice_probe evaluate --features artifacts/features --probes artifacts/probes \
  --out artifacts/evaluation
```

`train --answers ...` recomputes **all training-token states** for the final Ridge
fit, matching the reported fitting approach. Omit that argument for a quicker
sampled-prefix Ridge experiment; its fitting distribution differs from the table.
The compressor itself always trains on sampled prefixes. Selection uses validation
only. New question selection and Hugging Face generation will not exactly reproduce
the original cached trajectories or scores.

The reference collector is simple and serial; large-N collection is expensive.
Start small. See [the protocol and artifact guide](docs/USAGE.md) before scaling up.
You can supply your own numeric questions using the
[example JSONL format](examples/questions.jsonl).

## Predict from a prefix

After fitting a probe:

```bash
python -m slice_probe predict --checkpoint artifacts/probes/sparse-seed42.pt \
  --question 'There are 9 avocados and each serving needs 3. How many servings can be made?' \
  --prefix 'Each serving needs 3, so divide 9 by' --device auto
```

The output is a **raw correctness score** and its classification at threshold 0.5.
Ridge scores can fall outside `[0, 1]`; they are not certified probabilities or
proofs of correct reasoning. Use the same model, prompt and generation policy as
collection. Save both the checkpoint and its `.sha256.json` sidecar when moving
a fitted probe between machines.

## How SLICE learns

The reported Qwen2.5-Math-1.5B configuration uses blocks **28, 27, 25, 21, 13** (one-based, final first).
RMS-normalize each 1,536-vector and concatenate them. Training-only centering and
one isotropic scale fix the input coordinates.

A row-normalized projection plus ReLU maps 7,680 inputs into 768 features. A
temporary classifier trains those features with binary cross-entropy. Sparse
training adds `0.01 × mean(feature activation)`; plain training adds zero.

Discard the temporary classifier and freeze the compressor. Accumulate Ridge
statistics over training states and solve the regularized linear system. The
regularizer is `0.01 × I` on raw sums, including the intercept. Test predictions use
only causal prefixes. Eight states per retained answer include the question-only
state plus seven uniform nonterminal-prefix draws with replacement.

## Further reading

- Zhu et al., [The LLM Already Knows](https://arxiv.org/abs/2509.12886): expected answer quality from hidden states.
- Azaria & Mitchell, [The Internal State of an LLM Knows When It's Lying](https://arxiv.org/abs/2304.13734): hidden-state truthfulness probes.
- Kadavath et al., [Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221): self-evaluation and correctness estimation.
- [Qwen2.5-Math](https://arxiv.org/abs/2409.12122), [NuminaMath-1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5), and [Math-Verify](https://github.com/huggingface/Math-Verify): model, data and labels.

[MIT license](LICENSE) · [Citation](CITATION.cff)
