# Reproduce the corrected development check

Run commands from this directory. CPU only; no GPU is required. These are
same-machine macOS checks, not verified cross-platform installation claims.

## Environments

The corrected models/review ran with Python 3.12.4 and
`requirements-corrected.txt`. Accounting ran separately with Python 3.12.13 and
`requirements.txt` (including dp-accounting 0.6.0). A fresh combined environment
has not been installed or validated. Versions record the environments used for these experiments.

```bash
python -m pip install -r requirements-corrected.txt
python -m unittest discover -s tests -v
```

## Data and representation

Obtain `filtered_mmwave.zip` through the official MM-Fi toolbox download link,
then place it in `data/`. SHA-256:
`32942ec41f77634690a49ee4ba5d965c5df74e8842f90ef059aa37dd60b4418d`.
No raw data or processed subject features are redistributed.

```bash
python -m src.corrected_data
```

This rebuilds only corrected derived arrays and their audit. The immutable
historical adapter/results are separate. Only training/development payloads
are opened; existing test filenames may be enumerated without opening payloads.
The fixed split is `protocols/split.json`; don't alter it during reproduction.

## Reproduce existing results without overwriting them

```bash
python -m src.reproduce_corrected_references
python -m src.validate_corrected
```

The first command retrains all 12 references into a temporary output tree and
compares every metric with the saved CSV. The second independently retrains all
30 noisy/noiseless endpoints, checks masks and recomputes the paired interval.
It also checks the original source/protocol hashes. Existing fitted-result CSVs
are never overwritten. All seeds, including worse outcomes, remain in the CSVs.
Elapsed time is excluded from equality checks.

For a new research configuration, use a separate protocol and output namespace;
do not replace the recorded development outcomes. `src.corrected_models` and
`src.corrected_private` are the original fit entry points and refuse to overwrite
their final CSVs.

## Accounting, separate environment

```bash
python -m pip install -r requirements.txt
python -m src.corrected_account
```

The accountant recomputes the budget using integer RDP orders 2–128. It describes
one hypothetical mechanism with protected randomness. Public reproducibility
seeds in the utility runs do not provide an operational privacy guarantee.

## Notebook

`review.ipynb` reads saved CSV/JSON files and displays the results. Restart the
kernel and run all cells. Raw data are not required for this review-only route.
Execution does not retrain models. The displayed validation reports describe
previous actual reruns; use the commands above to perform them again.

## Historical adapter scope

The first adapter's failed results are retained separately in the research
archive. They are not substituted with the new recording-level scores. This
package contains the corrected development experiment and all its reported seeds.
`src/experiment.py` supplies shared gradient/probability helpers and historical
unit-test fixtures; the fit entry points for this package are `src.corrected_models`
and `src.corrected_private`. Do not run the historical fit entry point here.

## Rebuild the Appendix B figure

From the repository root in the corrected-model environment:

```bash
python -m src.appendix_visuals
```

This reads all 30 saved endpoints in `results/corrected/noise_runs.csv` and writes
`docs/figures/noise_paired_results.png` and `.pdf`. It checks complete paired seed
coverage before plotting. No data access or training is required. The notebook
calls the same plotting function. The paired t95% interval describes optimization
seeds on the fixed development cohort, not uncertainty over future subjects.
