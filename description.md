# 🌾 crop-loss-predictor-app

### ML-Powered Crop Loss Early Warning System for Ethiopia

HarvestGuard is a machine learning web application that predicts
the risk of post-harvest crop loss for smallholder farmers in
Ethiopia. Presumably, a user enters a region, crop type, and seasonal rainfall
and receives an instant High or Low risk prediction.

Built to support SDG 2 (Zero Hunger), SDG 9 (Innovation),
SDG 12 (Responsible Consumption), and SDG 13 (Climate Action).

## Problem
Ethiopian smallholder farmers lose 15–32% of crops post-harvest.
Warning signs are missed until it is too late to intervene.

## Solution
A FastAPI backend serves a trained models ex: Random Forest or XGBoost. A Streamlit
frontend lets NGO officers and ministry staff act before losses
occur — not after.

## Stack
- Model: scikit-learn · Random Forest · XGBoost · SHAP
- Backend: FastAPI → Render
- Frontend: Streamlit → Streamlit Cloud
- Data: World Bank Database · CHIRPS Rainfall

## Data
Two open CSV datasets merged on household_id, region + year:
LSMS (World Bank Microdata Library) and CHIRPS (Climate Hazards Group InfraRed Precipitation with Stations dataset, university of California, Santo Barbara) rainfall data.
No satellite image processing required.

## The Simple Summary

Farmers lose crops after harvest because of bad storage, moisture, insects, and slow transport — problems that are preventable if you know they are coming.

HarvestGuard predicts where and when those conditions are most dangerous — before the season peaks. The prediction goes to decision makers who can act fast enough to make a difference.

The model doesn't save the grain. The people who act on the prediction save the grain. The model just gives them enough warning to do so.

## sentences to add on my report's introduction from FAO:

"According to FAO & Ethiopian Statistics Service (2023), post-harvest losses for maize, wheat, faba bean and haricot bean in Ethiopia range from 8% to 17% nationally. This survey provides the empirical basis for our loss risk threshold definition."

That citation ads acadacim credibility without using FAO resource as training data. We will reference it, and don't model with it.

## For this project We will use LSMS (Living Standards Measurement Study) + CHIRPS datasets (Climate/Remote Sensing)

- LSMS (World Bank Microdata Library)
The five available waves cover 2011-12, 2013-14, 2015-16, 2018-19, and 2021-22, with Wave 5 released in 2024. There is no 2023, 2024 or 2025 data — the survey is implemented every two years and all data is made publicly available within twelve months of completion of each wave. The next wave would realistically be collected in 2023-24 and released around 2026-2027 at earliest.

- CHIRPS (https://data.humdata.org/dataset/eth-rainfall-subnational)
For CHIRPS: This is the good news. CHIRPS spans from 1981 to near-present — so rainfall data through 2024 absolutely exists. But without matching LSMS crop loss data for those same years, the extra rainfall years are useless for training.

- LSMS waves:
Wave 1: 2011-12 → ~3,969 households
Wave 2: 2013-14 → ~5,262 households
Wave 3: 2015-16 → ~5,469 households
Wave 4: 2018-19 → ~6,770 households
Wave 5: 2021-22 → ~4,999 households
─────────────────────────────────────
Total:           ~26,469 rows

## what we've built so far, across the two source datasets:

- 1. LSMS-ISA household survey data (5 waves: 2011, 2013, 2015, 2018, 2021)

Each wave came in a different raw format/questionnaire structure (CSV, SPSS, restructured section numbering across years), so for each one we had to verify which raw files actually held what (storage data, loss data, crop disposition, household roster), pick out only the relevant columns, and rename them to consistent, human-readable names.
Fixed a series of real data-quality issues along the way: memory blowups from SPSS category encoding, blank/missing household IDs, mismatched crop-identity fields, and (most recently) a household-ID precision bug that was silently corrupting 90% of Wave 2018's rows during the join to household data.

- 2. CHIRPS rainfall data

Downloaded dekadal (10-day) rainfall data per region from 1981–2026. We kept only region-level rows and the meaningful rainfall measures (10-day total, 1-month and 3-month totals, and each as a % of the long-term average — i.e. drought/flood signal).
Confirmed the CHIRPS region codes line up exactly with the LSMS region codes, so no manual lookup table was needed.
Aggregated the dekadal data up to Belg season, Meher season (Ethiopia's main growing season), and annual rainfall totals per region per year, then merged that onto crop_loss_master_all by region + year.
Where things stand now: crop_loss_master_all.csv has 94 columns — the crop-loss/storage/loss fields from all 5 waves, plus region_code/region_name, plus 6 rainfall features. It's a single file ready to be used for the prediction task. We also cleaned up a region-code inconsistency (mixed formats and, for one wave, region names instead of codes) that would have quietly broken the rainfall merge.

Everything is committed to notebooks/01_data_exploration.ipynb (which reproduces the whole pipeline end-to-end) and pushed to GitHub.

- 3. Feature engineering & the prediction task (notebooks/02_feature_exploration.ipynb)

Defined the actual prediction target: loss_occurred, a binary flag (True if any loss quantity was recorded for that household+crop+year, False otherwise) — matching the "High or Low risk" pitch above rather than regressing on loss quantity (those fields turned out to be in mixed, unnormalized units across reasons and waves, not safely comparable without extra work).

Along the way we caught and fixed leakage risks beyond the obvious one:
- Target leakage: total_loss_qty and every loss_reason*_occurred/unit/qty column are literally what the target is built from, so they can never be used as features — only to construct loss_occurred.
- Timing leakage: rainfall_annual_mm covers months after a typical Meher harvest — information a farmer wouldn't have yet at prediction time. Dropped it entirely, keeping only the seasonal Belg/Meher figures.
- Grouped leakage: a household can report multiple crops, so a random train/test split could put the same household on both sides and let a model partly memorize households instead of generalizing. Verified GroupShuffleSplit on household_id fixes it (the actual split itself is deliberately left for the modeling stage).

Also caught and fixed two duplicate-row bugs during a duplicate/missing-value audit: a household-ID float-precision bug (pandas silently promoted IDs to float64 during a read/concat, colliding distinct IDs) and ~1,934 rows that were only distinguished by an unreliable crop_code field (dropped entirely rather than patched, after finding 439 Wave-2013 rows where crop_code text disagreed with crop_name).

Where things stand now: `crop_loss_model_ready.csv` — 12 columns, 72,762 rows, 0 missing values, 0 duplicates, household_id still included and the data kept whole (not train/test split yet, by design). Target split: 93.3% no-loss / 6.7% loss — strongly imbalanced, so accuracy alone won't be a meaningful metric once training starts.

- 4. Modeling — baseline (notebooks/03_modeling.ipynb)

Split crop_loss_model_ready.csv with GroupShuffleSplit grouped on household_id (80/20, zero household overlap confirmed), then one-hot encoded crop_name/region_code/survey_year — fit on train only, so test-set categories can't leak into the encoding — giving 144 columns into a first RandomForestClassifier (class_weight="balanced" to counter the imbalance, chosen over Logistic Regression because the feature space has interactions we'd expect to be nonlinear). First-pass test-set result: ROC-AUC 0.750, loss-class precision 0.17 / recall 0.58 / F1 0.26.

Checked which features the model actually leans on using two different importance methods, and they disagreed. Scikit-learn's default (Mean Decrease in Impurity, training-data-based) ranked household_size far ahead of everything else — a known bias toward continuous/high-cardinality features. Permutation importance (test-set-based, unbiased by that) told a different story: household_size dropped to 7th, crop_name_TEFF became the top feature — and the seasonal rainfall features didn't crack the top 15 at all, surprising given the pitch above leans on seasonal rainfall as an input.

Chased that down rather than leaving it open. Two root causes, confirmed empirically: (1) rainfall is merged at region+year grain, so it's 100% determined by region_code+survey_year, which are also in the model; (2) even without survey_year, a region's rainfall only varies ~15% as much year-to-year as it varies between regions — so region_code alone (a real frontend input) already absorbs most of rainfall's signal. Dropping survey_year (which the frontend will never collect anyway — a live prediction's year is always unseen by the encoder) barely changed performance (ROC-AUC 0.750 → 0.748), confirming it wasn't doing meaningful work.

But "low average importance" turned out not to mean "rainfall doesn't matter." A synthetic sensitivity test — holding region and crop fixed, varying Meher rainfall from 40% to 160% of the region's historical average — showed predicted risk shifting by 8 to 57 percentage points depending on the scenario. One scenario swung wildly and non-monotonically, pointing at overfitting (the Random Forest had no max_depth/min_samples_leaf constraint, free to fit down to single-sample leaves). Adding min_samples_leaf=10 confirmed it: ROC-AUC rose to 0.808, recall rose to 0.73, and the erratic swing smoothed into a plausible shape. **This regularized model (rf_regularized, trained without survey_year) is now the official baseline** — strictly better than the first pass on every metric tracked.

One deliberate design decision worth recording here: Belg and Meher rainfall were kept as four separate columns (rainfall_belg_mm/_pct_of_avg, rainfall_meher_mm/_pct_of_avg) rather than merged into one rainfall figure. Ethiopia has two agriculturally distinct rainy seasons — Meher (Jun–Sep) is the main growing season for most crops, Belg (Feb–May) matters more for a smaller set of early-planted crops — so a single merged/summed number would blur that distinction and prevent the model from learning season-specific effects (e.g. that a poor Meher year hurts maize more than a poor Belg year does). A tree-based model like Random Forest can still learn crop×season interactions from the two separate columns without needing an explicit interaction feature. (The only rainfall figure that did get collapsed to a single annual number — rainfall_annual_mm — was dropped entirely instead of kept, for the timing-leakage reason above.)

- My colleague continues from here:
by adding an XGBoost model on the same X_train_no_year/X_test_no_year/y_train/y_test (not the original, superseded X_train/X_test) to compare against rf_regularized, using a consistent importance method (permutation or SHAP) rather than each model's own default, so the comparison is apples-to-apples. Not yet done: threshold tuning, an alternative imbalance strategy (e.g. SMOTE), a sample-size check for the thinner region×crop combinations, and a final model choice once XGBoost's results are in.

Everything is committed to notebooks/03_modeling.ipynb and pushed to GitHub.
