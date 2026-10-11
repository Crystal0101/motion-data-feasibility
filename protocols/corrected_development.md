# Corrected development experiment

This is an exploratory amendment after inspecting the first failed adapter, not a new independent confirmation. Historical results are retained. Test-subject payloads remain unopened. No hyperparameter search or model selection is performed in this amendment.

## Fixed representation and task

Use the existing 24 training and eight development subjects, all available activity recordings, 27 classes. One prediction per recording replaces the earlier window prediction. Scores from the two tasks must not be treated as an adapter-only improvement.

Keep the five binary columns anonymous (c1--c5); units and c4/c5 order are not established. Apply the same fixed monotone bounded transform 2 atan(v)/pi to each column, without fitting on any subject. Per observed frame compute five column means, standard deviations and 10/50/90 percentiles, plus count/(count+32) and payload-present=1 (including an empty payload, whose signal features are zero). Absent frames remain zero with payload-present=0. Empty payload counts are audited separately; the mask denotes availability, not nonempty signal. Preserve slots 1--297; do not concatenate across gaps. Aggregate consecutive blocks of ten slots (last block seven slots), averaging signal features over observed frames only and recording the observed fraction separately. Thus a recording has 30 steps and 27 features. These are index slots, not seconds. Upstream filtered frames already aggregate points; this is an offline benchmark, not causal acquisition.

For fixed-feature heads, use the temporal mean and standard deviation, last-half minus first-half mean, and centered-index covariance of those 27 block features (108 dimensions). A mask-only control uses the 30 observed fractions. These transforms have no fitted parameters.

## Non-private reference models

Seeds 41001--41003, all reported: logistic regression (C=1, max_iter=2000); ExtraTrees (250 trees, min_samples_leaf=2); GRU (27 inputs, 48 hidden units, linear 27-way readout, 40 epochs, Adam lr=0.003, batch size 32, no dropout). GRU uses the final hidden state, resets every recording, CPU two threads. GRU training uses the mean loss per recording without a weighted sampler. Evaluation gives each subject equal total confusion weight, and each recording equal weight within its subject; macro-F1 averages all 27 classes with zero-division defined as zero. Train/development metrics at epochs 0,5,...,40; endpoint is epoch 40, never best epoch. These are diagnostic references with different model families and budgets, not a controlled algorithm superiority comparison. Report training and development subject-balanced confusion/macro-F1. Mask-only logistic control checks acquisition-pattern confounding. A development F1 of 0.20 and reduction in training loss are engineering gates for useful learnability, not a validated clinical threshold. If the GRU gate fails, report failure without searching for a winning configuration.

## Fixed-feature noise simulation

Use only the fixed 108-dimensional transform, never the trained GRU encoder. Seeds 42001--42010; zero initialized softmax weights and bias, learning rate 1, 300 Poisson-sampled rounds q=1/3, fixed normalizer 8. Equal recording weight within each subject; concatenate all weight/bias gradients, then clip jointly at C=1. Compare unclipped, clipped noiseless and clipped Gaussian-noise simulation on identical subject masks and initialization. Include empty batches with noise. Calibrate one 300-round mechanism to epsilon<=3, delta=1e-5, add/remove-one subject using RDP orders 2--128. Save complete curves and all seed endpoints. The private comparison is paired; report mean difference and t interval over optimization seeds, not population uncertainty.

All random seeds and masks are public for reproducibility, so these outputs are noise utility simulations, NOT protected releases. No claim of joint privacy across runs, development metrics or encoder training. No SNN/local-rule efficacy or method novelty is tested. The tiny subject count is a limitation; failure does not prove DP is impossible.

## Validation

Checks cover monotonicity and noncollapsed features, missing-slot isolation,
held-out payload exclusion, subject disjointness, finite-difference gradients,
paired masks, complete seed coverage and source hashes. Historical outcomes and
the frozen CSDP study remain separate from the corrected experiment. The review
notebook displays saved metrics; rerun instructions are in `REPRODUCE.md`.
