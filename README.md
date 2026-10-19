# SLICE

**Sparse Latent Inference for Correctness Estimation**

SLICE predicts whether a language model's answer will be correct using the
question and an optional partial answer. It combines hidden states from a
**frozen language model**, compresses them into **768 learned features**, and
fits a **linear Ridge readout**. The language model stays frozen throughout.

![SLICE overview](docs/figures/overview.png)

## Results

Correctness prediction on 2,000 NuminaMath test questions using
**Qwen2.5-Math-1.5B**:

| Representation + readout | Features | Accuracy ↑ | AUROC ↑ | Brier ↓ |
|---|---:|---:|---:|---:|
| Constant training-prior baseline | — | 55.48% | 0.500 | 0.24705 |
| Final layer + Ridge | 1,536 | 66.89% | 0.737 | 0.21306 |
| Two-thirds layer + Ridge | 1,536 | 70.27% | 0.773 | 0.19702 |
| Five normalized layers + Ridge | 7,680 | 72.01% | 0.790 | 0.18917 |
| Plain bottleneck + Ridge | 768 | 72.63% | 0.800 | 0.18529 |
| **SLICE: sparse bottleneck + Ridge** | **768** | **72.66%** | **0.801** | **0.18505** |
| Five normalized layers + ten-layer residual probe | 7,680 | 73.30% | 0.804 | 0.17918 |

Using the **question alone**, SLICE reached **70.49% accuracy / 0.766 AUROC**.
These measure correctness prediction, not the language model's math-solving accuracy.
Plain and sparse bottlenecks perform nearly identically.

Compression reduces feature width **10×**, from 7,680 to 768. The Ridge readout
has 769 parameters; the compressor has about 5.9 million. This does not establish
an end-to-end speedup.

Results average three bottleneck-training seeds and condition on verifiable
answers. The test set was inspected during development. See
[study details](results/README.md) and [full metrics](results/reported-results.json).

## Quick start

Requires **Python 3.10+**.

```bash
git clone https://github.com/SupremeSingh/SLICE.git
cd SLICE
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
python -m pytest -q
python -m slice_probe demo --out artifacts/demo
```

The demo runs on CPU with synthetic features and no downloads. Its scores are
for checking the software. On Windows, activate with
`.venv\Scripts\Activate.ps1` instead.

For real model activations and answer collection, install the optional dependencies:

```bash
python -m pip install -e '.[models,data,verify]'
```

Commands default to `--device auto`, choosing CUDA, then Apple Silicon, then CPU.
Model and dataset downloads happen on first use. A GPU is recommended for larger runs.

## Try the sample

The included [sample](src/slice_probe/sample_answers.jsonl) contains seven authored
questions and fourteen labeled answers. This walkthrough downloads Qwen2.5-0.5B
and extracts real activations; it is a software example, not a benchmark.

```bash
python -m slice_probe sample --out artifacts/sample-answers
python -m slice_probe extract --answers artifacts/sample-answers --num-slices 3 \
  --out artifacts/sample-features
python -m slice_probe train --features artifacts/sample-features --width 32 --seeds 42 \
  --learning-rates 0.001 --epochs 5 --batch-size 32 --out artifacts/sample-probes
python -m slice_probe evaluate --features artifacts/sample-features \
  --probes artifacts/sample-probes --out artifacts/sample-evaluation
```

## Train a probe

Select questions, collect labeled answers, extract features, then train and evaluate:

```bash
python -m slice_probe prepare-numina --train 200 --val 50 --test 100 --out data/questions
python -m slice_probe collect --questions data/questions/questions.jsonl --responses 4 \
  --model Qwen/Qwen2.5-Math-1.5B --out artifacts/answers
python -m slice_probe extract --answers artifacts/answers --out artifacts/features
python -m slice_probe train --features artifacts/features --width 768 --seeds 42 \
  --out artifacts/probes
python -m slice_probe evaluate --features artifacts/features --probes artifacts/probes \
  --out artifacts/evaluation
```

This fits on sampled prefixes. Add `train --answers artifacts/answers` to fit the
final Ridge readout on all training-token states, as in the reported study.
Fresh runs use new data and should be reported separately from the table above.
Collection is serial and expensive at scale; start small.

To use your own questions, follow the [example JSONL format](examples/questions.jsonl).
See the [usage guide](docs/USAGE.md) for data requirements and fitting details.

## Compare methods

Run every baseline on the same sampled prefixes:

```bash
python -m slice_probe compare --answers artifacts/answers \
  --config configs/matched-prefix.json --out artifacts/comparison
```

This compares Ridge baselines, the residual probe, and plain/sparse bottlenecks.
It also tests whether prefix updates improve on question-only scores and rank
correct versus incorrect continuations of the same question. Calibration uses
validation data only.

See the [experiment guide](docs/REPRODUCIBILITY.md) for diagnostics, configuration
locks, calibration, and exporting review bundles.

## Predict from a prefix

```bash
python -m slice_probe predict --checkpoint artifacts/probes/sparse-seed42.pt \
  --question 'There are 9 avocados and each serving needs 3. How many servings can be made?' \
  --prefix 'Each serving needs 3, so divide 9 by'
```

Prediction returns a **raw correctness score** and a classification at threshold
0.5. Ridge scores can fall outside `[0, 1]`; probability estimates require the
optional validation-fitted calibrator. Use the same model, prompt, and generation
policy as training. Copy the checkpoint's `.sha256.json` sidecar when moving it.

## How it works

The reported model uses blocks **28, 27, 25, 21, 13** (one-based, final first).
Each 1,536-dimensional state is RMS-normalized, then concatenated into 7,680 features.
Training-only centering and one isotropic scale prepare the compressor input.

A row-normalized linear projection and ReLU produce 768 features. A temporary
classifier trains the compressor with binary cross-entropy; sparse training adds
`0.01 × mean(feature activation)`. The classifier is then replaced by a Ridge
readout fitted with the compressor frozen.

Completed answers receive Math-Verify labels. Each answer supplies a question-only
state and seven sampled nonterminal prefixes, all labeled by the final outcome.
Only the available prefix is visible to the predictor.

Use `--layer-base` and `--num-slices` with `extract` to change the selected blocks,
and `train --width` to change compression width. Preview a selection with:

```bash
python -m slice_probe layers --num-layers 28 --layer-base 2 --num-slices 5
# {"layers": [28, 27, 25, 21, 13]}
```

## Further reading

- Zhu et al., [The LLM Already Knows](https://arxiv.org/abs/2509.12886).
- Azaria & Mitchell, [The Internal State of an LLM Knows When It's Lying](https://arxiv.org/abs/2304.13734).
- Kadavath et al., [Language Models (Mostly) Know What They Know](https://arxiv.org/abs/2207.05221).
- [Qwen2.5-Math](https://arxiv.org/abs/2409.12122) · [NuminaMath-1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5) · [Math-Verify](https://github.com/huggingface/Math-Verify).

[MIT license](LICENSE) · [Citation](CITATION.cff)
