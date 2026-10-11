# Motion-data development supplement
Ning Yang

I corrected a saturated-column adapter and missing-index handling, then checked
whether MM-Fi radar activity recordings support a useful non-private reference.
The answer is yes for this development cohort. The fixed noisy-head configuration
still fails its utility check. This is independent of the frozen CSDP appendix;
it does not demonstrate a new SNN method or a successful private model.

Start with [review.ipynb](review.ipynb). It reads saved results, includes all ten
noise-comparison seeds, their paired differences and all three GRU learning curves. Core logic is in
`src/`; the notebook serves as the presentation interface. Full reruns and
same-machine checks are documented in [REPRODUCE.md](REPRODUCE.md).

## Correction and evidence

The original fourth-column scale clipped all inspected values. I now retain all
five columns anonymously and apply the same fixed monotone transform to each.
No physical units or c4/c5 swap are assumed. Missing index slots stay in place,
with an explicit availability mask. See [DATA_VERIFICATION.md](DATA_VERIFICATION.md).

The new task predicts one label per recording, using 646 training recordings
from 24 people and 215 development recordings from eight different people. It is
not directly comparable to the original window-level task. The eight held-out
subjects' sensor payloads remain unopened. Filtered source frames are already
aggregated; the benchmark does not establish online causality.

| Diagnostic reference | Mean development macro-F1 | Seed range |
|---|---:|---:|
| Fixed-feature logistic regression | 54.08% | deterministic repeats |
| Fixed-feature ExtraTrees | 69.28% | 68.85–69.71% |
| GRU, fixed epoch 40 | 56.80% | 53.81–61.33% |
| Availability-mask-only logistic regression | 4.04% | deterministic repeats |

These families have different training budgets. The table checks learnability,
not algorithm superiority. Evaluation uses a subject-balanced confusion matrix
and all 27 classes. Three GRU and tree seeds are preserved; deterministic logistic
repeats are not three independent experiments. No best-epoch score is reported.

## The noise result is negative

The fixed-feature softmax head averages 46.47% macro-F1 without noise and 1.89%
with the prescribed noise. The paired difference over ten seeds is −44.59 pp,
with t95% interval [−45.60, −43.57] pp. This interval concerns optimization
randomness on a fixed development cohort, not uncertainty over future people.

One hypothetical 300-round Poisson Gaussian mechanism has RDP-accounted
ε≤3, δ=10⁻⁵, q=1/3, joint subject clip C=1, noise multiplier 8.738442 and fixed
normalizer eight. It uses the fixed representation, never the trained GRU.
Public seeds and masks make the published-style artifacts **utility simulations,
not protected releases**. This single-run account does not cover multiple runs,
private encoder fitting or development-metric releases.

Noise changes the optimization trajectory: clipping is inactive in noiseless
controls but affects about 89.84–96.62% of selected noisy-run updates. This study
does not isolate the cause of the loss or establish an optimal clipping policy.
Failure at this fixed configuration does not prove subject-level DP impossible.

## What is verified

- Sixteen tests pass, including finite-difference subject gradients, joint
  clipping, empty batches, missing-slot isolation and noncollapsed features.
- A separate mathematical implementation retrained all 30 noise-comparison
  endpoints; saved accuracy and F1 matched exactly.
- All 12 diagnostic reference fits were repeated on the same machine, matching
  all numerical metrics exactly (elapsed time excluded).
- The English notebook executes from a fresh kernel with consecutive code-cell
  counts and no error outputs. No raw data are embedded or redistributed.

This is an exploratory amendment after earlier development inspection, not an
independently preregistered confirmation. Protocol and training-source hashes were
saved before the fits; the budget hash preceded the noise simulations. The earlier
failed adapter results are retained in the research archive. Current per-seed
results and the revised protocol are included here.

[Appendix B](docs/appendix_motion.pdf) accompanies the proposal. Its reference [24]
refers to Yang et al. (2023), linked below. The original CSDP appendix is a separate
[archived release](https://github.com/Crystal0101/csdp-local-goodness/releases/tag/v1.0-proposal-appendix).

## Remaining research

A compact local-learning SNN, a defensible privacy/utility design and a genuinely
held-out subject comparison remain to be tested under a separate protocol.
The current results do not establish local-learning superiority, energy savings,
fall detection or clinical benefit. Units, true timing and upstream aggregation
direction remain unverified. The fixed transform avoids inventing those facts;
it does not resolve them.

## Sources

- Yang et al. (2023), *MM-Fi: Multi-Modal Non-Intrusive 4D Human Dataset for
  Versatile Wireless Sensing*, NeurIPS Datasets and Benchmarks.
  [Dataset page](https://ntu-aiot-lab.github.io/mm-fi),
  [official toolbox](https://github.com/ybhbingo/MMFi_dataset/tree/269f64875508f030cbd1468c1e76d5333b0127cd).
- [Official filtering notice](https://docs.google.com/document/d/1484VK1Lw7JWQ6FHqIEypx0sZ3xzw2MeG7QhRvA5ihoc/edit).
- [Google differential-privacy accounting](https://github.com/google/differential-privacy/tree/main/python/dp_accounting), v0.6.0.
