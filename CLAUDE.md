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
  13-column feature set, leakage/unreliable columns dropped, missing values
  resolved, 0 duplicates, `household_id` still included and the data still
  whole (not train/test split — see "Prediction task" below for why the
  split is deliberately deferred to model-training time).
- `model/`, `backend/`, `frontend/` — empty so far, not yet started.
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
  separate PyPI package for these.
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
  no-loss. Class split is **93.5% no-loss / 6.5% loss** — strongly
  imbalanced; any model needs explicit handling (class weighting,
  resampling, or a metric other than plain accuracy) rather than being
  trained naively.
- **`total_loss_qty` itself is built from `loss_reason1/2/3_qty`** (summed
  across up to 3 reported loss reasons per household+crop, after first
  summing across that household's parcels/fields for the same crop). Because
  of this, `total_loss_qty` and every `loss_reason1/2/3_occurred/unit/qty`
  column are **target leakage** — they directly encode the outcome and must
  never be used as model features, only to construct the target.
- **Feature set (`df_model`, 14 columns after all cleanup)** after dropping
  leakage and unreliable columns: `household_id` (grouping key, not a
  feature), `crop_code`/`crop_name` (code is the model input, name is
  display), `household_size`, `region_code`/`region_name` (same split),
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
  `rainfall_annual_mm`/`_pct_of_avg` (timing leakage — see below).
- **Missing values**: the 381-row household-join-gap cluster is **dropped**
  (imputing region/rainfall for a row where it's genuinely unknown would
  inject a wrong value into what's otherwise a real predictive signal).
  `crop_code`/`crop_name` missing independently are **recoded to an explicit
  "Unknown" category** (`-1`/`"UNKNOWN"`) rather than mode-imputed or
  dropped, to avoid biasing toward the most common crop and to keep those
  rows' `loss_occurred` label. Final `df_model` after this: 74,696 rows, 0
  missing values.
- **Duplicate check caught a real bug** (see the household-ID gotcha below) —
  0 duplicates now, both full-row and on the natural
  `household_id`+`crop_code`+`crop_name`+`survey_year` grain.
- **Small sample sizes**: `region_name` (10 values) and `survey_year` (5
  values) are low-cardinality with thousands of rows each, but `crop_name`
  has 189 distinct values and 69 of them have fewer than 30 rows — a loss
  rate computed on that few rows is mostly noise given the 6.5% base rate.
  Not fixed at the data-prep stage; a model needs to either group rare crops
  into an "other" bucket or accept it can't make a confident crop-specific
  call for them.
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
- The notebook can grow too large for the `Read` tool once it's been executed
  (outputs embedded). If `Read`/`NotebookEdit` fail on size, edit the
  underlying `.ipynb` JSON directly with a small Python script
  (`json.load` → mutate `cell["source"]` as a list of lines, clear
  `cell["outputs"]`/`execution_count` on code cells → `json.dump`), then
  re-run via `jupyter nbconvert --to notebook --execute --inplace`.
