# Break-aware CausalImpact intervals: one pipeline for every table

Data: Kaggle `mikhail1681/walmart-sales`, saved as `Walmart_Sales.csv` (45 stores, first 143 weeks). Not redistributed.

## One pipeline
`unified_pipeline.py` produces the tables and figures of Sections 8.1-8.15 from one data loader, one scenario
generator (7 scenarios), one diagnostic (year-over-year), one split protocol (22/23 stores, 20 splits), one set of
lambda/k grids and one threshold rule. Only the BSTS engine changes, via `--engine`:

| Engine | What it is | Cost |
|---|---|---|
| `kalman` (default) | local-level BSTS, maximum likelihood, numpy/scipy only, deterministic | minutes |
| `tfci` | tfcausalimpact 0.0.19 (variational inference) | hours; needs TensorFlow |

```
pip install -r requirements.txt
python unified_pipeline.py --data Walmart_Sales.csv --out results                       # all 7 scenarios, kalman
python unified_pipeline.py --data Walmart_Sales.csv --out results_tfci --engine tfci \
       --scenarios none,common5                                                          # same protocol, other engine
```
Compare `results/` and `results_tfci/` on `none`/`common5` to see how much of any difference is the engine alone.
The script stops if the real CSV is missing. `--smoke-test` uses synthetic data, writes to `*_SMOKE_SYNTHETIC`,
stamps the figures, and is for testing the code only.

## Outputs (in `--out`)
`all_results.csv` (split x scenario x method), `split_params.csv`, `table_coverage_pct_*`, `table_width_vs_standard_*`,
`table_mae_*`, `table_bias_*` (`primary_split` and `mean_over_splits`; Tables 8.10-8.13 layout), `table_8_7_detector_comparison.csv`,
`table_8_7b_pooled_controls_detection.csv`, `table_early_warning_summary.csv`, `table_detection_power_by_shift_size.csv`,
`table_8_8_point_correction.csv`, `extras_bootstrap_and_pooled_width.csv`, `fig8_9_scenarios.png`, `fig8_7_calibration.png`,
`run_config.json` (engine, grids, data SHA-256).

## Changes that alter numbers relative to the old code
* Diagnostic in the 7-scenario runs is now the notebook's YoY statistic (the old script used a different regression version).
* Grids are the union of the two old grids; the selected lambda/k can therefore differ from both old runs.
* Point correction uses expm1(S) x seasonal-naive post level (notebook definition).
* The old `load_real` ignored its `path` argument and read a hard-coded Windows path; fixed.
* Table 8.7b (pooled controls) code was not in the old repository; it is reconstructed from the paper's text
  (|mean z of K controls - calibration mean| > t(.975, n-1) * s * sqrt(1/K + 1/n); six draws per test store). Check it
  against the original code before relying on it.

## Status
Run on the real Walmart file with `--engine kalman` (20 splits, about 11 seconds); results are in `results_real/`.
Consistency check: the engine-independent control diagnostic reproduces the paper's Table 8.7 exactly (35.43% detection of a
5% shift at 5.00% false alarms, AUC 0.81, primary-split threshold 7.58). The pooled-controls table (8.7b) is reconstructed and gives
63.6/83.8/97.5% for 3/5/10 controls against the paper's 62.9/83.4/96.8%; the paper's values were kept.
The `tfci` engine has only been exercised against a mock package; Sections 8.1-8.11 have not yet been regenerated with the unified pipeline.
`run-shift.py` and the notebook are superseded.
