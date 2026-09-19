# HarvestGuard — Crop Loss Predictor (Ethiopia)

ML app predicting post-harvest crop loss in Ethiopia from LSMS-ISA/ESS
household survey data + CHIRPS rainfall data. See `description.md` for the
project pitch/motivation.

## Repo layout

- `notebooks/01_data_exploration.ipynb` — the single source of truth for the
  entire data pipeline (raw files → `data/processed/*.csv`). It's meant to be
  runnable top-to-bottom (`jupyter nbconvert --to notebook --execute --inplace
  notebooks/01_data_exploration.ipynb`) and reproduce every processed file
  from scratch. Parts, in order: Part 1-2 build Wave 1 (2011), Part 3 builds
  Wave 2 (2013), Part 4 builds Wave 3 (2015), Part 5 builds Waves 4 & 5
  (2018/2021, one shared function), Part 6 combines all 5 into
  `crop_loss_master_all.csv`, Part 7 explores the combined file, Part 8
  explores/selects CHIRPS rainfall columns, Part 9 aggregates rainfall to
  season/year and merges it into `crop_loss_master_all.csv`.
- `notebooks/02_feature_exploration.ipynb` — starts from the finished
  `crop_loss_master_all.csv` (no raw-data processing here). Loads the data,
  audits missingness, defines the `loss_occurred` binary target, drops
  leakage/unreliable columns, and leaves a clean `df_model` feature set. See
  "Prediction task" below for the details this notebook established.
- `notebooks/03_modeling.ipynb` — starts from `crop_loss_model_ready.csv`.
  Grouped train/test split (`GroupShuffleSplit` on `household_id`, 80/20,
  `random_state=42` — `train_df`/`test_df`, 58,122 / 14,366 rows).

  **Official baseline: `rf_regularized`** — `RandomForestClassifier`
  (`class_weight="balanced"`, `min_samples_leaf=10`, `random_state=42`)
  trained on `X_train_no_year`/`X_test_no_year` (one-hot `crop_name` +
  `region_code` only — `survey_year` deliberately excluded, see step 4
  below — 117 columns). Test-set result: **ROC-AUC 0.810**, loss-class
  precision 0.16 / recall 0.72 / F1 0.26. This supersedes an earlier, worse
  baseline (`rf`: same `class_weight`, unregularized, trained with
  `survey_year` included — ROC-AUC 0.748, recall 0.58) — kept in the
  notebook as part of the investigation trail below, not the model to
  build on. (All `03` numbers here are after `02`'s crop-name spelling
  merge; before it, on 123 crops, the same models scored 0.808 / 0.750.)

  **How we got here** (each step is its own titled section in the
  notebook, in order):
  1. Fit `rf` on `X_train`/`X_test`/`y_train`/`y_test` (122 columns,
     one-hot `crop_name`/`region_code`/`survey_year`) as the first
     baseline — ROC-AUC 0.748.
  2. Two feature-importance methods disagreed sharply: scikit-learn's
     default MDI (training-data-based) ranked `household_size` first
     (0.30), but permutation importance (test-set-based, unbiased by
     cardinality) dropped it to #7 and ranked `crop_name_TEFF` first
     instead — MDI is known to inflate continuous/high-cardinality
     features. **Permutation importance is the trustworthy one.**
  3. Rainfall didn't crack permutation importance's top 15 at all, which
     was surprising given the pitch leans on seasonal rainfall. Root-caused
     via two checks: (a) rainfall is merged at region+year grain, so it's
     100% determined by `region_code`+`survey_year` (confirmed via a
     groupby-nunique check); (b) a `region_code`-alone redundancy check
     showed each region's rainfall varies only ~15% as much year-to-year as
     it varies *between* regions, so `region_code` (a genuine frontend
     input) already absorbs most of rainfall's signal — permutation
     importance wasn't wrong, rainfall's *marginal* contribution really is
     small on average.
  4. Ablation: refit without `survey_year` (which the frontend will never
     collect — any live year is unseen by the encoder and gets zeroed out
     by `handle_unknown="ignore"`) — performance barely moved (ROC-AUC
     0.748 → 0.751), confirming `survey_year` wasn't doing meaningful work
     and can be dropped for free. This produced `X_train_no_year`/
     `X_test_no_year` (117 columns), the feature set the official baseline
     uses.
  5. A synthetic sensitivity scenario (hold region+crop fixed, vary Meher
     rainfall 40%-160% of the region's long-term average) showed rainfall
     *does* move predictions meaningfully (9-61 percentage points
     depending on the scenario) — low average importance and a large local
     effect aren't contradictory. But one scenario (Benishangul-Gumuz +
     Teff) swung wildly and non-monotonically (0.65 → 0.04 → 0.20),
     suggesting overfitting to a sparse region×crop×rainfall slice, since
     the model had no `max_depth`/`min_samples_leaf` constraint.
  6. Regularizing with `min_samples_leaf=10` confirmed the overfitting
     diagnosis: ROC-AUC rose to 0.810, recall rose to 0.72, and the wild
     swing smoothed into the 0.58-0.70 range — this became
     `rf_regularized`, the official baseline above.

  `rf_regularized` is now superseded by `04_xgboost.ipynb`'s XGBoost model
  (below). It stays the reference point every `04` comparison is made
  against, and `04` asserts it still scores ROC-AUC 0.810.
- `notebooks/04_xgboost.ipynb` — XGBoost vs Random Forest, tuning, and the
  **final model choice**. Rebuilds `03`'s exact split and 117-column
  no-`survey_year` features (asserted by row/column counts). Runs ~30 min on
  an M2 (mostly Optuna); keep the machine awake, since a sleeping Mac pauses
  the kernel. Ground rules: the test set is scored once (section 13), and
  every decision is made with **grouped 5-fold CV on the training set**
  (`StratifiedGroupKFold` on `household_id`, the same 5 folds reused for
  every experiment). The headline metric is **PR-AUC** (average precision),
  with ROC-AUC kept for continuity with `03`. **Decision rule used
  throughout**: extra complexity is adopted only if it wins PR-AUC in all 5
  folds (a sign test, ~1-in-32 by chance). A first full run used a "mean
  gain > 0.005" rule instead, which dropped `household_size` at +0.0047
  even though XGBoost's own +0.0042 win was being accepted. That was
  inconsistent, so it was replaced and re-run (documented in section 10).

  **Final model: XGBoost** (`n_estimators=121`, `learning_rate` 0.094,
  `max_depth` 8, `gamma` 9.35, `reg_lambda` 15.6, `colsample_bytree` 0.45,
  `scale_pos_weight` 13.95, full settings in `model/model_card.json`), on the
  same 8 inputs as `rf_regularized`, threshold **0.498** (highest threshold
  with ≥75% recall on out-of-fold predictions). Test: **ROC-AUC 0.814, PR-AUC
  0.231** (baseline 0.810 / 0.223), recall 77.3% (754/976 losses) at 15.6%
  precision (baseline at 0.5: 72.4% / 16.0%). CV PR-AUC 0.222 vs 0.212
  baseline, 0.218 for an equally tuned RF (50 Optuna trials each). XGBoost
  beats the tuned RF in all 5 folds and on test ROC-AUC in 100% of 1,000
  household-bootstrap resamples; the test PR-AUC gap is within noise (82%).
  At equal 75% recall, precision is 15.9% vs 15.7%: **a real but small
  gain; the inputs, not the algorithm, are the ceiling.**

  Decisions made in CV (sections 8-10): `scale_pos_weight` beat no weighting
  and SMOTE-NC in every fold (so **scores are not probabilities**: mean score
  0.38 vs 0.067 loss rate, and the app must show High/Low, not "x% chance");
  grouping 25 rare crops into `OTHER (RARE)` changed nothing (crops stay
  separate); **`household_size` stays** (helps in 5/5 folds, +0.0047 PR-AUC).

  Other findings: rainfall alone flips High/Low for 26 of 140 common
  region+crop combinations (19%); crop identity dominates both SHAP and
  grouped permutation importance (~5-8× the next input). **`crop_name =
  UNKNOWN` rows (1,925, all no-loss) are a recording artifact** that gives
  every model easy correct answers; `04` section 13 re-scores without them
  (`test_set_known_crops_only` in the model card). 339 of 579 region×crop
  combinations have <30 training rows → a `limited_data` flag in
  `model/region_crop_support.csv`.
  **Not yet done**: probability calibration; the backend itself.
- `data/raw/` — gitignored, not tracked. Contains one folder per LSMS wave
  (`ETH_2011_ERSS_v02_M_CSV`, `ETH_2013_ESS_v03_M_SPSS`,
  `ETH_2015_ESS_v03_M_CSV`, `ETH_2018_ESS_v04_M_CSV`,
  `ETH_2021_ESPS-W5_v02_M_CSV`) plus `data/raw/CHIRPS/` (dekadal rainfall CSV
  from data.humdata.org). If raw data is ever missing, the notebook will fail
  at whichever wave's raw folder is gone — that's expected, not a bug to
  "fix" in the code; restore the raw folder instead.
- `data/processed/` — tracked in git. `crop_loss_master_2011.csv` ...
  `_2021.csv` (one per wave) plus `crop_loss_master_all.csv` (all 5 waves
  concatenated, with rainfall merged in) and `rainfall_region_dekadal.csv`
  (intermediate CHIRPS output, region-level, still dekadal grain,
  pre-seasonal-aggregation). **`crop_loss_model_ready.csv`** (built at the
  end of `02_feature_exploration.ipynb`) is the actual file to start
  modeling from — `crop_loss_master_all.csv` reindexed down to the clean
  12-column feature set, leakage/unreliable columns dropped, missing values
  resolved, crop-name spelling variants merged (101 crops), re-aggregated
  to one row per `household_id`+`crop_name`+`survey_year` (72,488 rows),
  0 duplicates, `household_id` still included
  and the data still whole (not train/test split — see "Prediction task"
  below for why the split is deliberately deferred to model-training time).
- `model/` — written by `04_xgboost.ipynb` section 18.
  `harvestguard_xgb.joblib` is a scikit-learn `Pipeline` (one-hot encoding +
  XGBoost) refit on all 72,488 rows. It takes raw columns in
  `model_card.json`'s `input_column_order` and returns a score to compare
  against `decision_threshold`. It's **gitignored** (`*.joblib`), so rebuild
  it by running `04`. `model_card.json` (inputs and where each comes from,
  threshold, settings, CV/test scores, library versions) and
  `region_crop_support.csv` (rows and loss rate per region+crop, plus a
  `limited_data` flag for <30 rows) are tracked.
- `backend/`, `frontend/` — empty so far, not yet started.
  **Frontend input design, decided ahead of building it**: not every
  feature the model needs should be a manual input field. `crop_name` and
  `region_code` (dropdown, or GPS resolved to a region server-side) are the
  genuine user inputs. **`household_size` is also asked**: `04` measured it
  (section 10) and it improves the model in every CV fold. Dropping it
  would cost ~2% relative PR-AUC for a shorter form, a product call that
  can be reversed by flipping `KEEP_HOUSEHOLD_SIZE` in `04` and re-running. `is_rural` should be hardcoded/defaulted, not asked —
  the app's whole audience is smallholder farmers, so it's ~always 1.
  **The four rainfall columns (`rainfall_belg_mm`/`_pct_of_avg`,
  `rainfall_meher_mm`/`_pct_of_avg`) must never be manual entry fields** —
  no farmer knows exact seasonal rainfall in mm or % of average. The
  FastAPI backend should fetch current-season rainfall automatically for
  the selected region (from CHIRPS or a similar live weather API, the same
  source the training data came from) once crop+region are chosen, and
  compute those four features server-side before calling the model. This
  shapes the backend's request schema, so decide it before that endpoint is
  built, not after.
- `.kiro/steering/` — pulls `CLAUDE.md` in as Kiro's project memory
  (`project-context.md`) plus a `workflow.md` with environment/git
  conventions, so Kiro and Claude share one memory file instead of drifting
  apart if this project is worked on in both tools. Keep editing `CLAUDE.md`
  as the canonical source; update `workflow.md` too if a convention below
  changes (e.g. the environment name).

## Environment

- Dedicated pyenv-virtualenv **`harvestguard`** (Python 3.12.9), set via
  `pyenv local harvestguard` (`.python-version` in the repo root, auto-
  activates on `cd`). Deliberately separate from the shared `lewagon`
  bootcamp environment used for other coursework, so this project's
  dependencies (`shap`, `fastapi`, `streamlit`, none of which `lewagon` had)
  can't drift into or get broken by unrelated assignments.
- Dependencies are pinned in `requirements.txt` (`pip install -r
  requirements.txt`) — exact versions, not ranges, since this pipeline has
  already been bitten twice by subtle version-dependent behavior (the SPSS
  categorical-groupby OOM, the household-ID float-precision bug). Keep it in
  sync with whatever's actually installed. `scikit-learn` alone covers
  Random Forest / Logistic Regression / Gradient Boosting — there's no
  separate PyPI package for these. On macOS, `xgboost` also needs the
  OpenMP runtime from Homebrew (`brew install libomp`) or it fails on
  import.
- Migrating to `harvestguard` was verified safe: both notebooks were re-run
  end to end under it with zero errors, and every `data/processed/*.csv`
  came out byte-identical to the `lewagon`-produced versions.

## Data pipeline — key facts and decisions

- **5 survey waves, 5 different raw formats.** Each wave's questionnaire was
  restructured (file names, section numbers, and which file holds what all
  changed across waves — do not assume Wave N's file layout matches Wave 1's
  without checking). This was verified file-by-file per wave; see the
  notebook's markdown cells for what was actually confirmed vs assumed per
  wave.
- **Schema harmonization**: `BASE_SCHEMA` is captured live from Wave 1's
  finished output columns (Part 2), and every later wave's output is
  reindexed to that same column set before saving. A wave-specific column
  that doesn't exist in another wave shows up as all-`NaN` there — that's
  expected, not a bug.
- **`_unconfirmed` suffix convention**: columns renamed from raw codes where
  the exact semantics couldn't be verified against an official codebook (no
  official LSMS codebook was available during this build). Treat these as
  lower-confidence features; don't assume the name is a precise description
  of what the question actually asked.
- **Disposition data (what happened to the harvest) is empty for Waves 4/5
  (2018, 2021) by deliberate decision** — their raw disposition section
  (`sect11`) couldn't be reliably linked to a specific crop (its own
  crop-name field is mostly garbled text, and the crop-id it does have is
  scoped per-field, not per-holder). Storage and loss data for those two
  waves is fine and crop-identified; only disposition/`crop_domain` is blank.
- **`region_code` is the canonical join/model key across all 5 waves**,
  normalized to one clean nullable-int dtype in Part 6. `region_name` is a
  separate, display-only column (mapped from `region_code`) — added because
  the raw source spellings genuinely disagree across files (e.g. Wave 2's
  SPSS labels literally spell it "Somalie"/"Diredwa"/"Benshagul Gumuz").
  **General rule for this project**: numeric codes are the source of truth
  for joins/training; human-readable labels are a lookup applied only at the
  display layer (e.g. in a future mobile UI). Don't merge or train on text
  labels.
- **`region_code` 14 (Addis Ababa) never appears** in any wave's crop-loss
  data — expected, not a bug. It's a purely urban region with no
  agricultural households in the post-harvest module.
- **CHIRPS region `PCODE` numeric suffix == LSMS `region_code` directly**
  (confirmed via Wave 2's SPSS value labels for `saq01`, e.g. region_code 14
  ↔ `ET14`). No separate crosswalk table was needed. CHIRPS zone-level data
  (`adm_level == 2`) was NOT used — LSMS's `zone_code` is a per-region local
  sequence, not nationally unique, so it doesn't safely match CHIRPS zone
  PCODEs without extra work.
- **Rainfall is merged at region + calendar-year grain**, aggregated into
  Belg (Feb-May), Meher (Jun-Sep, Ethiopia's main growing season), and annual
  totals + % of long-term average. This is joined to `crop_loss_master_all`
  by matching `survey_year` to the rainfall's calendar year — an unconfirmed
  assumption (LSMS doesn't expose exact interview dates in the columns kept
  here) that each wave's crop-loss rows report on that same calendar year's
  Meher season. Sanity-checked once: 2015 shows 88% of normal Meher rainfall,
  consistent with Ethiopia's known 2015 El Niño drought.

## Prediction task (established in `02_feature_exploration.ipynb`)

- **Binary classification**, matching `description.md`'s "High or Low risk"
  pitch — not regression. A regression on `total_loss_qty` (actual quantity
  lost) was considered and set aside: loss quantities are in mixed,
  unnormalized units across reasons/waves, not reliable for regression
  without unit harmonization first (unsolved).
- **Target**: `loss_occurred` = `(total_loss_qty.fillna(0) > 0)`. `NaN` in
  `total_loss_qty` means "no loss entry recorded for this crop," treated as
  no-loss. Class split is **93.3% no-loss / 6.7% loss** in the final
  `crop_loss_model_ready.csv` (93.5%/6.5% before the `crop_code` cleanup and
  grain re-aggregation below nudged it slightly) — strongly imbalanced; any
  model needs explicit handling (class weighting, resampling, or a metric
  other than plain accuracy) rather than being trained naively.
- **`total_loss_qty` itself is built from `loss_reason1/2/3_qty`** (summed
  across up to 3 reported loss reasons per household+crop, after first
  summing across that household's parcels/fields for the same crop). Because
  of this, `total_loss_qty` and every `loss_reason1/2/3_occurred/unit/qty`
  column are **target leakage** — they directly encode the outcome and must
  never be used as model features, only to construct the target.
- **Feature set (`df_model`, 12 columns after all cleanup)** after dropping
  leakage and unreliable columns: `household_id` (grouping key, not a
  feature), `crop_name` (the sole crop-identity feature — see why
  `crop_code` was dropped entirely, below), `household_size`,
  `region_code`/`region_name` (code is the model input, name is display),
  `is_rural`, `survey_year`, `rainfall_belg_mm`/`_pct_of_avg`,
  `rainfall_meher_mm`/`_pct_of_avg`, and `loss_occurred`. All ≤5.2% missing
  before cleanup (`crop_name` highest at 5.2%; most of the rest were the same
  ~381 rows from the known household-join gap, dropped — see below).
- **Dropped and why**: `loss_detail_*`/`loss_extra_*` (70-99.9% missing, some
  describe the loss circumstance itself → leakage risk too) · all
  `storage_*_unconfirmed` (69-98% missing, too sparse to trust) ·
  `disposition_*` (45-46% missing, absent entirely for Waves 2018/2021,
  ambiguous timing relative to the loss event) · `crop_domain` (35% missing,
  redundant with crop identity) · `zone_code`/`woreda_code` (zone_code is a
  per-region local sequence, not nationally unique/comparable as-is) ·
  `ph_saq07`/`ph_saq07_loss` (82-84% missing, semantics unconfirmed) ·
  `rainfall_annual_mm`/`_pct_of_avg` (timing leakage — see below) ·
  **`crop_code` (dropped entirely, not just missing-value handled — see the
  gotcha below; it isn't numeric for Wave 2013, and 439 of that wave's rows
  have `crop_code` text that outright disagrees with `crop_name`)**.
- **Missing values**: the 381-row household-join-gap cluster is **dropped**
  (imputing region/rainfall for a row where it's genuinely unknown would
  inject a wrong value into what's otherwise a real predictive signal).
  `crop_name` missing independently (5.2%) is **recoded to an explicit
  "Unknown" category** (`"UNKNOWN"`) rather than mode-imputed or dropped, to
  avoid biasing toward the most common crop and to keep those rows'
  `loss_occurred` label.
- **Duplicate check caught two real bugs**, not one:
  1. See the household-ID float-precision gotcha below — fixed upstream in
     `01_data_exploration.ipynb`.
  2. Dropping `crop_code` (above) exposed ~1,934 rows that used to be
     distinguished *only* by `crop_code` (e.g. two differently-coded plots of
     the same named crop for one household) — almost all exact copies once
     `crop_code` is gone, but 6 rows (3 pairs) had a genuine conflict: same
     household+crop+year+rainfall context, but `loss_occurred` disagreed (1
     vs 0). **Fixed by re-aggregating** `df_model` to one row per
     `household_id`+`crop_name`+`survey_year`: covariates via `"first"`
     (verified exactly constant within each group — they only depend on
     `household_id`+`survey_year`, never on the dropped `crop_code`) and
     `loss_occurred` via `"max"` (did this household have *any* loss on this
     crop that year). The same step also collapses the rows that the
     crop-name spelling merge (see the gotcha below) turned into
     duplicates. Together these took `df_model` from 74,696 down to its
     final 72,488 rows (5 groups with conflicting `loss_occurred`, resolved
     by `"max"`). 0 duplicates now, both full-row and on the
     `household_id`+`crop_name`+`survey_year` grain.
- **Small sample sizes**: `region_name` (10 values) and `survey_year` (5
  values) are low-cardinality with thousands of rows each, but `crop_name`
  has 101 distinct values (189 before the Wave 2021 double-prefix bug was
  fixed, 123 before the spelling-variant merge — see the gotchas below) and
  21 of them have fewer than 30 rows — a loss rate computed on that few rows
  is mostly noise given the 6.7% base rate.
  Not fixed at the data-prep stage. `04_xgboost.ipynb` tested grouping rare
  crops into an "other" bucket (see its section 9) and handles thin
  region×crop combinations with a `limited_data` flag instead.
- **Two leakage risks beyond target leakage, both resolved**:
  1. *Rainfall timing vs. the app's actual use*: `rainfall_annual_mm` covers
     the full calendar year including Oct-Jan, the window *after* a typical
     Meher harvest — i.e. during/after the loss event the app is meant to
     warn about beforehand. ~16% of the annual total falls in that
     post-harvest window, and it's only moderately correlated with the
     Belg+Meher seasonal total (r=0.70/0.45), so it's real extra information
     a farmer wouldn't have yet at prediction time — not just a restatement
     of the seasonal figures. Also matches `description.md`'s own pitch,
     which describes the input as "seasonal rainfall," not annual.
     **Dropped `rainfall_annual_mm`/`_pct_of_avg` from `df_model` entirely.**
  2. *Grouped leakage from `household_id`*: a household can report multiple
     crops, so a naive random train/test split could put the same
     household's rows on both sides, letting a model partly recognize a
     specific household instead of learning a generalizable pattern.
     Verified in `02_feature_exploration.ipynb` (now removed from that
     notebook after verifying) that `sklearn.model_selection.GroupShuffleSplit`
     grouped on `household_id` (80/20, `random_state=42`) gives zero
     `household_id` overlap and a loss rate that stays close to 6.5%/6.6% in
     both halves (not accidentally skewed by the grouping). **The actual
     split is deliberately deferred to model-training time, not done in this
     EDA/feature-prep notebook** — `crop_loss_model_ready.csv` is saved
     whole, `household_id` included, ready to be split when training starts.
     Plain `sklearn.model_selection.train_test_split` cannot be used for that
     split as-is — it has no `groups` parameter, so a naive call would
     reintroduce this exact leakage; use `GroupShuffleSplit` (or dedupe to
     unique `household_id`s, split those, then filter rows by the result).

## Known gotchas when touching this pipeline

- `pandas.read_spss` auto-decodes SPSS value labels to text — grouping by a
  resulting `Categorical` column with `dropna=False` enumerates every
  category combination (including unused ones) and can OOM. Cast to plain
  strings before grouping.
- Watch for blank (`""`, not `NaN`) `household_id` values in raw exports —
  seen in Wave 2's SPSS files. These must be filtered out before any
  `dropna=False` groupby or they silently collapse into one bogus household.
- Don't normalize long integer ID columns (household IDs are ~17-18 digits)
  through `pd.to_numeric()` — it can silently return `float64` for the whole
  column and lose precision past 2^53, corrupting most IDs. Normalize IDs as
  strings only (strip whitespace, drop a stray trailing `.0`, strip leading
  zeros) — never round-trip them through a numeric/float dtype. This bug
  class has bitten the pipeline **twice** from two different directions: once
  during Waves 4/5's raw-data join (fixed with the string normalization
  above), and once via `pd.read_csv` itself — a single raw row in Wave 3
  (2015)'s loss file had a missing `household_id`, which forced pandas to
  infer `float64` for that whole column on read; dropping the bad row
  afterward didn't undo the dtype, and that `float64` column then forced
  Part 6's `pd.concat` to promote the *entire* combined `household_id`
  column to `float64` on every wave, silently colliding distinct
  Wave-2018/2021 household IDs that happened to be numerically close (e.g.
  sequential within the same enumeration area). Showed up as ~740 false
  "duplicate" rows in `02_feature_exploration.ipynb`'s grain-duplicate check
  before being traced back and fixed. **Takeaway**: any `pd.read_csv` that
  touches `household_id` should pass `dtype={"household_id": str}`
  explicitly — don't rely on there happening to be zero nulls at read time to
  keep it as an integer type.
- **`crop_code` is unreliable and not used as a feature — unresolved
  upstream issue, documented but not fixed.** For Wave 2013 specifically,
  `crop_code` in `crop_loss_master_all.csv` is not a numeric code at all —
  it's crop-name text (18,302 of 18,305 Wave-2013 rows), because Wave 2013's
  raw data only had a decoded crop-name field, backfilled into both
  `crop_code` and `crop_name` (`01_data_exploration.ipynb` Part 3). Worse,
  **439 of those rows have `crop_code` text that disagrees with
  `crop_name`** (e.g. `crop_code="HARICOT BEANS"` next to
  `crop_name="CACTUS"` on the same row) — a real construction bug in that
  wave's build, not just inconsistent formatting. `02_feature_exploration.ipynb`
  sidesteps this by dropping `crop_code` entirely and using `crop_name` as
  the sole crop-identity feature, rather than fixing the root cause (which
  would mean re-examining Wave 2013's `sect9a`/`sect10`/`sect11`/`sect12`
  merge logic in `01_data_exploration.ipynb`). If `crop_code` is ever needed
  again (e.g. to cross-reference an official LSMS crop-code list), this
  Wave 2013 issue needs solving first.
- **Wave 2021's raw crop-label field can double the numeric prefix** — e.g.
  `s9q00b` = `"2. 2.MAIZE"` instead of the normal `"2. MAIZE"` (confirmed in
  `sect9_ph_w5.csv`). `split_code_label()`'s original regex only stripped one
  `"N."` prefix, so 8,188 rows (~11% of the full dataset) ended up with
  `crop_name` values like `"2.MAIZE"` instead of `"MAIZE"` — fragmenting 66
  crops into two categories each (e.g. `"MAIZE"` and `"2.MAIZE"` both
  existed). Caught via the one-hot-encoded column count in
  `03_modeling.ipynb` looking too high, traced back to `01`. **Fixed**: the
  regex now strips a repeated `"N."` prefix
  (`r"^\s*(\d+)\.\s*(?:\d+\.\s*)*(.+?)\s*$"`, verified against every known
  label format before applying). Distinct `crop_name` values dropped from
  189 to 122 after re-running the full pipeline.
- **The same crop is spelled differently across waves** — even after the
  prefix fix above. Wave 2021 drops the space in multi-word names
  (`CHICKPEAS` vs `CHICK PEAS`, `SUGARCANE` vs `SUGAR CANE`, 14 pairs in
  all), and some labels were reworded or cut off between waves
  (`OTHER ROOT C` / `OTHER ROOT CROP` / `OTHER ROOT CROPS`; three spellings
  of Mung bean; `NUEG` vs `NUEG OR NIGERSEED`; `WHITE LUMIN` vs
  `WHITECUMIN`). Each group was confirmed to be one crop by its shared
  numeric `crop_code` outside Wave 2013. **Fixed in
  `02_feature_exploration.ipynb`** with an explicit `CROP_NAME_FIXES`
  dictionary (123 → 101 crops), not in `01`: 01 needs `data/raw/`, which
  isn't in every clone, so `crop_loss_master_all.csv` still carries the raw
  spellings. The cell asserts that no two remaining names differ only by
  spacing, punctuation or case, so a new variant from a future re-run of 01
  fails loudly instead of slipping through. A handful of rows (1-5 per code)
  carry a numeric `crop_code` that belongs to a different crop than their
  `crop_name` — data-entry noise, left as is.
- The notebook can grow too large for the `Read` tool once it's been executed
  (outputs embedded). If `Read`/`NotebookEdit` fail on size, edit the
  underlying `.ipynb` JSON directly with a small Python script
  (`json.load` → mutate `cell["source"]` as a list of lines, clear
  `cell["outputs"]`/`execution_count` on code cells → `json.dump`), then
  re-run via `jupyter nbconvert --to notebook --execute --inplace`.
