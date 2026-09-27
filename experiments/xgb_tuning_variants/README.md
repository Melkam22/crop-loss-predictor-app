# Experiment: can XGBoost get honestly better?

**Short answer: not XGBoost itself, but a new feature helps both models.**
The historical loss rate per region+crop pair (target-encoded) lifts Random
Forest in all 10 fresh folds. With that feature, XGBoost and Random Forest are
tied again (PR-AUC 0.2264 vs 0.2259). See "Part 2" below.

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
  better algorithm". Part 2 checks this.

## Part 2: Random Forest gets the same feature (`rf_region_crop.py`)

Same method: Optuna on `04`'s 5 folds using `04`'s own RF search space, each
study started from `04`'s tuned RF, then confirmed on the **same** 10 fresh
folds. That makes the results pair fold-by-fold with Part 1. Test set still
untouched.

| RF variant | Trials | PR-AUC | ROC-AUC | Folds beating tuned RF (no feature) | Folds XGBoost's best beats it |
|---|---|---|---|---|---|
| RF + region×crop (one-hot, ~700 extra columns) | 20* | 0.2237 ± 0.014 | 0.7956 | 7/10 | 7/10 |
| **RF + target-encoded rates** | 50 | **0.2259 ± 0.013** | **0.8009** | **10/10** | 7/10 (mean gap +0.0005) |

\* ~3 min per trial with the extra columns, so it got a smaller budget. Its
best was still `04`'s starting settings, so more trials were unlikely to help.

Best RF + target-encoded settings: `max_depth=23`, `min_samples_leaf=18`,
`max_features=0.2`, `class_weight="balanced_subsample"`, smoothing `m≈82`.

## Overall conclusion

- **The feature is the real finding.** Adding a smoothed historical loss rate
  for each crop and each region+crop pair beats `04`'s tuned RF in **all 10**
  fresh folds (0.2224 → 0.2259 PR-AUC, ROC-AUC 0.7945 → 0.8009). It is the
  only change in this experiment that passes the project's all-folds rule.
- **The algorithm is not.** With the feature, XGBoost's best (0.2264) and RF
  (0.2259) are tied, a 0.0005 gap that is within noise. XGBoost's apparent
  lead in Part 1 came from the feature, not from XGBoost.
- **Cost of adopting it**: the backend would need a lookup table of those
  loss rates (per crop, per region+crop), saved next to the model and built
  from training data only. `model/region_crop_support.csv` already stores
  a raw version of this. No new form fields.

## Decision needed (not taken yet)

- **Option A: keep the old way.** `04`'s XGBoost as-is, no new feature.
- **Option B: adopt the feature.** Add the target-encoded rates to `04` as a
  new section, pick RF or XGBoost (tied, so the simpler one to ship), score
  that one finalist on the test set once, then rebuild `model/` and update
  `CLAUDE.md`.

## Files

- `xgb_variants.py`: the full experiment (search + confirm). Run from the
  repo root with the `harvestguard` env:
  `python experiments/xgb_tuning_variants/xgb_variants.py [out.json] [n_trials]`.
  With 50 trials it takes ~45 min on an M2; keep the Mac awake
  (`caffeinate -i python ...`).
- `results.json`: per-fold PR-AUC/ROC-AUC for both stages, plus each
  variant's best hyperparameters (`best_params`).
- `run_log.txt`: console output of the run above.
- `rf_region_crop.py`: Part 2, the Random Forest fairness check (~63 min).
  Run after `xgb_variants.py`, since it reads `results.json` to compare
  fold by fold.
- `results_rf.json` / `run_log_rf.txt`: Part 2's results and console output.
