# 🌾 HarvestGuard

### ML-Powered Crop Loss Early Warning System for Ethiopia

HarvestGuard is a machine learning web application that predicts
the risk of post-harvest crop loss for smallholder farmers in
Ethiopia. A user enters a region, crop type, and seasonal rainfall
— and receives an instant High or Low risk prediction.

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
- Data: FAO Ethiopia · CHIRPS Rainfall

## Data
Two open CSV datasets merged on region + year:
FAO post-harvest loss survey and CHIRPS rainfall data.
No satellite image processing required.
