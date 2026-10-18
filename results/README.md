# Reported frozen-model results

`reported-results.json` is a portable metric export from the completed Numina
probe study. It includes per-head metrics, calibration bins, prefix diagnostics,
question-level paired intervals, data counts, size accounting and a cached-feature
CPU microbenchmark. No model weights, individual question texts or large caches
are included. The top-level README rounds these values for readability.

The model was Qwen2.5-Math-1.5B. The main experiment generated 104,000 answers,
retained 102,329, and evaluated 125,920 sampled prefixes from 15,740 retained
answers on 2,000 test questions. Plain and sparse values average three
feature-training seeds on shared data. They are not ensemble scores or independent
training-corpus replications.

The reported final Ridge fit used all training-token states; compressor training
and test evaluation used sampled prefixes. The residual-network reference trained
on sampled prefixes. Differences between those head families also change state
weighting. The inspected test set is conditional on verifiable answers; unknown
semantic/pretraining overlap and automatic verification errors remain limitations.
Question-only predictions repeat a question representation against its sampled
outcomes, not against independent new questions.

Sparse-768 accuracy exceeds final-layer Ridge by about 5.8 percentage points and
two-thirds Ridge by about 2.4 points. Both paired question accuracy intervals
exclude zero descriptively. The sparse-versus-plain accuracy interval includes
zero, while its Brier difference is small. The deeper residual probe has better
Brier and calibration. There is no superiority claim over every predictor.

The compressor contains 5,899,008 projection parameters and the Ridge readout 769.
Cached-feature CPU compression-plus-readout time was approximately 0.20 ms per
prefix, amortized over batch size 256 with eight CPU threads. This excludes the
LLM, capture, transfers and I/O. Direct layer readouts were faster in this kernel
benchmark. The sparse and plain implementations both use dense operations.
A smaller feature vector establishes neither end-to-end speedup nor guaranteed
probability calibration.
