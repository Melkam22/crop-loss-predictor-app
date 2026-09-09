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
- `data/raw/` — gitignored, not tracked. Contains one folder per LSMS wave
  (`ETH_2011_ERSS_v02_M_CSV`, `ETH_2013_ESS_v03_M_SPSS`,
  `ETH_2015_ESS_v03_M_CSV`, `ETH_2018_ESS_v04_M_CSV`,
  `ETH_2021_ESPS-W5_v02_M_CSV`) plus `data/raw/CHIRPS/` (dekadal rainfall CSV
  from data.humdata.org). If raw data is ever missing, the notebook will fail
  at whichever wave's raw folder is gone — that's expected, not a bug to
  "fix" in the code; restore the raw folder instead.
- `data/processed/` — tracked in git. `crop_loss_master_2011.csv` ...
  `_2021.csv` (one per wave) plus `crop_loss_master_all.csv` (all 5 waves
  concatenated, with rainfall merged in — this is the file to use for
  modeling) and `rainfall_region_dekadal.csv` (intermediate CHIRPS output,
  region-level, still dekadal grain, pre-seasonal-aggregation).
- `model/`, `backend/`, `frontend/` — empty so far, not yet started.

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

## Known gotchas when touching this pipeline

- `pandas.read_spss` auto-decodes SPSS value labels to text — grouping by a
  resulting `Categorical` column with `dropna=False` enumerates every
  category combination (including unused ones) and can OOM. Cast to plain
  strings before grouping.
- Watch for blank (`""`, not `NaN`) `household_id` values in raw exports —
  seen in Wave 2's SPSS files. These must be filtered out before any
  `dropna=False` groupby or they silently collapse into one bogus household.
- Don't normalize long integer ID columns (household IDs are ~17 digits)
  through `pd.to_numeric()` — it can silently return `float64` for the whole
  column and lose precision past 2^53, corrupting most IDs. Normalize IDs as
  strings only (strip whitespace, drop a stray trailing `.0`, strip leading
  zeros) — never round-trip them through a numeric/float dtype.
- The notebook can grow too large for the `Read` tool once it's been executed
  (outputs embedded). If `Read`/`NotebookEdit` fail on size, edit the
  underlying `.ipynb` JSON directly with a small Python script
  (`json.load` → mutate `cell["source"]` as a list of lines, clear
  `cell["outputs"]`/`execution_count` on code cells → `json.dump`), then
  re-run via `jupyter nbconvert --to notebook --execute --inplace`.
