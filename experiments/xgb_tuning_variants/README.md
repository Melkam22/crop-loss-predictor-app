# Experiment: can XGBoost get honestly better?

**Status: experimental. Nothing here is adopted.** `notebooks/04_xgboost.ipynb`,
`model/` and the official baseline are unchanged. This folder is a side
experiment, kept separate so it can't disturb the main pipeline.

## Why

After the `crop_name = UNKNOWN` rows were dropped, `04`'s tuned XGBoost and
`03`'s `rf_regularized` ended up essentially tied on the test set (PR-AUC
0.2056 vs 0.2066). This experiment asks whether better tuning or better
features give XGBoost a real edge, **without** fooling ourselves.

## How it was kept honest

1. **Same training data as `04`**: same grouped 80/20 split
   (`GroupShuffleSplit`, `random_state=42`, 56,395 training rows). The
   **test set was not touched at all.**
2. **Search**: each variant got its own Optuna study (50 trials, PR-AUC
   objective) on `04`'s exact 5 grouped folds (`StratifiedGroupKFold`,
   `random_state=42`). Every study started from `04`'s winning settings.
3. **Confirm**: the best setting of every variant was then re-scored on
   **10 fresh grouped folds** (two new splits, `random_state=7` and `123`)
   that no search ever saw. Picking the best of 50 trials always looks a bit
   better than it really is. The fresh folds remove that luck, and they are
   the numbers to trust. Stage 1 scores ran about 0.006 PR-AUC higher than
   stage 2 for every variant.
4. Early stopping always used an inner 10% household split of the fitting
   data, never the fold being scored. Target encodings (variant D) were
   computed out-of-fold inside the fitting data, so a row never sees its own
   label.

## Variants tried

| Variant | What changes |
|---|---|
| One-hot, wider search | `04`'s features; lower learning rates allowed, up to 3000 trees, `scale_pos_weight` tuned too |
| Native categorical | XGBoost's own categorical splits for `crop_name`/`region_code` instead of one-hot |
| Native + region×crop | as above, plus one extra categorical: the region+crop pair (e.g. `3\|TEFF`) |
| One-hot + target-encoded rates | `04`'s one-hot, plus smoothed historical loss rate per crop and per region×crop |
| Native + target-encoded rates | native categoricals plus the same two rates |
| 5-seed average | the best variant, averaged over 5 random seeds |

## Results (10 fresh folds, mean ± std)

| Model | PR-AUC | ROC-AUC | Folds beating tuned RF |
|---|---|---|---|
| RF baseline (`03`'s `rf_regularized`) | 0.2175 ± 0.014 | 0.7923 | 1/10 |
| RF tuned (`04`) | 0.2224 ± 0.013 | 0.7945 | — |
| XGBoost tuned (`04`, current model) | 0.2209 ± 0.015 | 0.7931 | 4/10 |
| One-hot, wider search | 0.2228 ± 0.015 | 0.7964 | 6/10 |
| Native categorical | 0.2192 ± 0.019 | 0.7929 | 3/10 |
| **Native + region×crop** | **0.2264 ± 0.018** | **0.7995** | **7/10** |
| One-hot + target-encoded rates | 0.2258 ± 0.017 | 0.7989 | 6/10 |
| Native + target-encoded rates | 0.2229 ± 0.021 | 0.7964 | 4/10 |
| Native + region×crop, 5-seed average | 0.2267 ± 0.017 | 0.8000 | 8/10 |

## What it means

- **The gain comes from a feature, not from tuning.** A wider search alone
  barely moved the score, and native categoricals alone were slightly worse.
  What helped is telling the model about each region+crop *pair* directly.
  Both top variants do this, either as a category or as a loss rate.
- **The gain is real but small**: about +0.004 PR-AUC over tuned RF
  (~2% relative) and +0.0055 over the current XGBoost. It wins 7-8 of 10
  folds, not all 10, so it **does not pass the project's decision rule**
  (win PR-AUC in every fold).
- **No new app inputs needed.** Region×crop is built from crop and region,
  which the form already collects.
- **Not yet a fair fight**: Random Forest was never given the region×crop
  feature. Part or all of this gain might be "a better feature", not "a
  better algorithm".

## Open next steps

1. Give tuned RF the same region×crop feature and re-check on the fresh folds.
2. If XGBoost still leads, score that one finalist on the test set once.
3. Only then fold it into `04` (as a new section), rebuild `model/`, and
   update `CLAUDE.md`.

## Files

- `xgb_variants.py`: the full experiment (search + confirm). Run from the
  repo root with the `harvestguard` env:
  `python experiments/xgb_tuning_variants/xgb_variants.py [out.json] [n_trials]`.
  With 50 trials it takes ~45 min on an M2; keep the Mac awake
  (`caffeinate -i python ...`).
- `results.json`: per-fold PR-AUC/ROC-AUC for both stages, plus each
  variant's best hyperparameters (`best_params`).
- `run_log.txt`: console output of the run above.
