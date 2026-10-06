# 🌾 HarvestGuard — Crop Loss Predictor

ML-powered early warning for post-harvest crop loss in Ethiopia. Pick a
region and crop, enter household size, and get a High or Low risk rating —
seasonal rainfall is fetched and computed automatically, never a manual
input.

### 🔗 [Try it live](https://crop-loss-predictor-app-uww3xptjpuw4rmqtn8mkapp.streamlit.app/)

Built on five waves of World Bank LSMS-ISA household surveys (2011–2021)
plus CHIRPS satellite rainfall data, in support of SDG 2 (Zero Hunger).

## How it's built

- **Model**: XGBoost, tuned and compared against a Random Forest baseline
  with household-grouped cross-validation. Catches ~7 in 10 real losses on
  unseen households (recall 0.726).
- **Backend**: FastAPI (`backend/`), Dockerized, deployed on
  [Render](https://render.com). Fetches and computes live rainfall
  features from CHIRPS server-side.
- **Frontend**: Streamlit (`frontend/`), deployed on Streamlit Community
  Cloud, calling the backend's public API.
- **Data pipeline**: `notebooks/01_data_exploration.ipynb` through
  `04_xgboost.ipynb` — reproducible end to end from raw survey files to the
  deployed model.

See [`CLAUDE.md`](CLAUDE.md) for full technical documentation (repo layout,
data pipeline decisions, modeling journey, backend/deployment details) and
[`description.md`](description.md) for the project's pitch and narrative.

## Team

Built at Tomorrow University of Applied Sciences by Ashenafi Shiferaw
(idea & data), Azmain Morshed (tech & engineering), and Fatih Vardar
(business & presentation).
