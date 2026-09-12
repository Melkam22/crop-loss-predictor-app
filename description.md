# 🌾 crop-loss-predictor-app

### ML-Powered Crop Loss Early Warning System for Ethiopia

HarvestGuard is a machine learning web application that predicts
the risk of post-harvest crop loss for smallholder farmers in
Ethiopia. A user enters a region, crop type, and seasonal rainfall
and receives an instant High or Low risk prediction.

Built to support SDG 2 (Zero Hunger), SDG 9 (Innovation),
SDG 12 (Responsible Consumption), and SDG 13 (Climate Action).

## Problem
Ethiopian smallholder farmers lose 15–32% of crops post-harvest.
Warning signs are missed until it is too late to intervene.

## Solution
A FastAPI backend serves a trained XGBoost model. A Streamlit
frontend lets NGO officers and ministry staff act before losses
occur — not after.

## Stack
- Model: scikit-learn · XGBoost · SHAP
- Backend: FastAPI → Render
- Frontend: Streamlit → Streamlit Cloud
- Data: World Bank Database · CHIRPS Rainfall

## Data
Two open CSV datasets merged on household_id, region + year:
LSMS (World Bank Microdata Library) and CHIRPS rainfall data.
No satellite image processing required.

## The Simple Summary

Farmers lose crops after harvest because of bad storage, moisture, insects, and slow transport — problems that are preventable if you know they are coming.

HarvestGuard predicts where and when those conditions are most dangerous — before the season peaks. The prediction goes to decision makers who can act fast enough to make a difference.

The model doesn't save the grain. The people who act on the prediction save the grain. The model just gives them enough warning to do so.

## sentences to add on my report's introduction from FAO:

"According to FAO & Ethiopian Statistics Service (2023), post-harvest losses for maize, wheat, faba bean and haricot bean in Ethiopia range from 8% to 17% nationally. This survey provides the empirical basis for our loss risk threshold definition."

That citation adds academic credibility without requiring us to use FAO as training data. We will reference it, and don't model with it.

## For this project We will use LSMS + CHIRPS datasets

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
