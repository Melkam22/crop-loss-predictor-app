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

Later, a crop-name spelling clean-up merged variants of the same crop across waves (e.g. `CHICKPEAS` in 2021 vs `CHICK PEAS` in earlier waves, three spellings of mung bean), taking crop_name from 122 to 100 distinct crops. Each merge was confirmed by the crops sharing the same numeric crop code in the raw survey.

One more reversal worth recording: `crop_name`-missing rows (5.2%) were originally recoded to an explicit `"UNKNOWN"` category rather than mode-imputed or dropped, on the assumption the missingness was random. My colleague's XGBoost work in `04_xgboost.ipynb` found that assumption false — all 1,925 `UNKNOWN` rows have `loss_occurred = 0`, a recording artifact (no loss can be logged against an unnamed crop) rather than independent noise. The model had learned `UNKNOWN = no loss` as its single strongest signal, and since the app's frontend never sends `UNKNOWN` (a user always selects a real crop), keeping those rows just inflated every reported score with a case the app will never see live. Those rows are now dropped instead of recoded.

Where things stand now: `crop_loss_model_ready.csv` — 12 columns, 70,563 rows, 0 missing values, 0 duplicates, household_id still included and the data kept whole (not train/test split yet, by design). Target split: 93.1% no-loss / 6.9% loss (up slightly from 93.3%/6.7% before the `UNKNOWN` drop, since those rows were 100% no-loss by construction) — strongly imbalanced, so accuracy alone won't be a meaningful metric once training starts.

- 4. Modeling — baseline (notebooks/03_modeling.ipynb)

Split crop_loss_model_ready.csv with GroupShuffleSplit grouped on household_id (80/20, zero household overlap confirmed), then one-hot encoded crop_name/region_code/survey_year — fit on train only, so test-set categories can't leak into the encoding — giving 121 columns into a first RandomForestClassifier (class_weight="balanced" to counter the imbalance, chosen over Logistic Regression because the feature space has interactions we'd expect to be nonlinear). First-pass test-set result: ROC-AUC 0.720, loss-class precision 0.15 / recall 0.54 / F1 0.23.

Checked which features the model actually leans on using two different importance methods, and they disagreed. Scikit-learn's default (Mean Decrease in Impurity, training-data-based) ranked household_size far ahead of everything else — a known bias toward continuous/high-cardinality features. Permutation importance (test-set-based, unbiased by that) told a different story: household_size dropped out of the top 15 entirely, crop_name_TEFF became the top feature — and the seasonal rainfall features didn't crack the top 15 at all, surprising given the pitch above leans on seasonal rainfall as an input.

Chased that down rather than leaving it open. Two root causes, confirmed empirically: (1) rainfall is merged at region+year grain, so it's 100% determined by region_code+survey_year, which are also in the model; (2) even without survey_year, a region's rainfall only varies ~15% as much year-to-year as it varies between regions — so region_code alone (a real frontend input) already absorbs most of rainfall's signal. Dropping survey_year (which the frontend will never collect anyway — a live prediction's year is always unseen by the encoder) barely changed performance (ROC-AUC 0.720 → 0.722), confirming it wasn't doing meaningful work.

But "low average importance" turned out not to mean "rainfall doesn't matter." A synthetic sensitivity test — holding region and crop fixed, varying Meher rainfall from 40% to 160% of the region's historical average — showed predicted risk shifting by 4 to 55 percentage points depending on the scenario. One scenario swung wildly and non-monotonically, pointing at overfitting (the Random Forest had no max_depth/min_samples_leaf constraint, free to fit down to single-sample leaves). Adding min_samples_leaf=10 confirmed it: ROC-AUC rose to 0.792, recall rose to 0.70, and the erratic swing smoothed out. **This regularized model (rf_regularized, trained without survey_year) is now the official baseline** — clearly better than the first pass on ranking quality and recall, at similar precision. (Numbers in this section are after dropping the `crop_name = UNKNOWN` rows, described above; on the prior data — crop-name clean-up applied, `UNKNOWN` still kept — the same model scored ROC-AUC 0.810.)

One deliberate design decision worth recording here: Belg and Meher rainfall were kept as four separate columns (rainfall_belg_mm/_pct_of_avg, rainfall_meher_mm/_pct_of_avg) rather than merged into one rainfall figure. Ethiopia has two agriculturally distinct rainy seasons — Meher (Jun–Sep) is the main growing season for most crops, Belg (Feb–May) matters more for a smaller set of early-planted crops — so a single merged/summed number would blur that distinction and prevent the model from learning season-specific effects (e.g. that a poor Meher year hurts maize more than a poor Belg year does). A tree-based model like Random Forest can still learn crop×season interactions from the two separate columns without needing an explicit interaction feature. (The only rainfall figure that did get collapsed to a single annual number — rainfall_annual_mm — was dropped entirely instead of kept, for the timing-leakage reason above.)

- My colleague continued from here in `notebooks/04_xgboost.ipynb` (section 6 below).

- 5. Frontend/backend input design (decided ahead of time, not yet built)

Before backend/frontend work starts: not every feature the model needs should be a manual input field. crop_name and region (dropdown or GPS-resolved) are the two genuine user inputs. household_size was a judgment call, since it wasn't in the original pitch's described inputs. It was settled by measurement in section 6: it improves the model in every cross-validation fold, so the form asks for it. is_rural should just be hardcoded/defaulted, since the app's whole audience is smallholder farmers — asking it would be pointless.

The four rainfall columns (rainfall_belg_mm/_pct_of_avg, rainfall_meher_mm/_pct_of_avg) should never be manual entry fields — no farmer knows exact seasonal rainfall figures in mm or % of average. Instead, the FastAPI backend should fetch current-season rainfall automatically for the selected region (from CHIRPS or a similar live weather API, the same source the training data came from) once crop+region are chosen, and compute those four features server-side before calling the model. This changes what the backend's request schema looks like, so it's worth deciding now rather than after the endpoint is built.

- 6. XGBoost vs Random Forest — tuning, comparison, and a result that changed (notebooks/04_xgboost.ipynb)

My colleague built XGBoost on the exact same split and features as rf_regularized, and gave both algorithms the same tuning effort (50 Optuna trials each, scored by 5-fold cross-validation grouped by household on the training set only), so the comparison is tuned vs tuned. The headline metric moved from ROC-AUC to PR-AUC (average precision), which judges a model by the quality of its flagged list, the part an early-warning tool actually uses, and suits a low base rate better. ROC-AUC is still reported throughout. The test set was scored once, at the end. Every choice before that (settings, imbalance strategy, threshold) came from cross-validation, with one rule for "is this difference real": the better option must win in all 5 folds.

What was decided, each in cross-validation: class weighting (scale_pos_weight) beat both no weighting and SMOTE-NC oversampling in every fold. Grouping 25 rare crops into one "other" category made no difference, so crops stay separate. household_size improved the model in every fold, so it stays. The "High risk" threshold is the highest one that still catches at least 75% of real losses on out-of-fold predictions.

The original result (on the data before the crop_name = UNKNOWN fix in section 3): XGBoost was the clear final model, test ROC-AUC 0.814 vs 0.810 and PR-AUC 0.231 vs 0.223, beating the equally tuned Random Forest in all 5 cross-validation folds. My colleague's own notebook flagged the UNKNOWN rows as a caveat inflating both models' scores similarly — but after actually dropping those rows in 02 and re-running 03 and 04 end to end, the picture changed more than "similarly inflated" suggested: **XGBoost and rf_regularized are now essentially tied.** Test ROC-AUC 0.7923 vs 0.7924, PR-AUC 0.2056 vs 0.2066, precision 0.1388 vs 0.1416 — XGBoost is marginally behind on all three, and ahead only on recall (72.6% vs 70.0%, catching 18 more of the 902 real losses at this threshold). So the earlier "XGBoost wins" conclusion was substantially an artifact of the UNKNOWN rows, not a real algorithmic edge; what's left is a modest recall argument for XGBoost, not a clear win. This correction is recorded as a new section 20 in the notebook, added without rewriting my colleague's original sections 1-19.

The rest of my colleague's findings aren't affected by this and still stand: the inputs set the ceiling, not the algorithm. Crop identity carries most of the signal (SHAP and permutation importance agree, several times the next input), and rainfall only has a few dozen distinct region-year values to learn from. Rainfall still matters: on its own it flips the High/Low verdict for a meaningful share of common region+crop combinations. The scores aren't probabilities (class weighting inflates them), so the app must show High/Low, not "x% chance". Region+crop combinations with fewer than 30 training rows get a "limited data" note (model/region_crop_support.csv).

Where things stand now: the model is saved in model/ (a scikit-learn pipeline, rebuilt by running the notebook, plus a model card with inputs, threshold and scores). Given the corrected comparison above, whether XGBoost or the Random Forest baseline ships in the FastAPI backend is a closer call than the notebook's original write-up suggests — worth a deliberate decision, not just deferring to whichever notebook ran last.
