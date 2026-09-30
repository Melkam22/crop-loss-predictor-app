import os
from pathlib import Path

# Defaults assume the Docker Compose layout: WORKDIR /app, with model/ mounted
# read-only at /app/model and data/raw/CHIRPS mounted read-write at
# /app/data/raw/CHIRPS. Override via env vars for local (non-Docker) runs.
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/app/model"))
MODEL_PATH = MODEL_DIR / "harvestguard_xgb.joblib"
MODEL_CARD_PATH = MODEL_DIR / "model_card.json"
REGION_SUPPORT_PATH = MODEL_DIR / "region_crop_support.csv"

CHIRPS_CSV_PATH = Path(
    os.environ.get("CHIRPS_CSV_PATH", "/app/data/raw/CHIRPS/eth-rainfall-subnat-full.csv")
)

# Direct HDX resource-download URL (copied by hand from the dataset's Download
# button, not the CKAN JSON API -- that endpoint returned 403 when tested).
# Stable: redirects to a freshly pre-signed S3 URL on every request.
CHIRPS_RESOURCE_URL = os.environ.get(
    "CHIRPS_RESOURCE_URL",
    "https://data.humdata.org/dataset/423143be-315f-48d7-9e90-ae23738da564/"
    "resource/49e3a707-d153-423e-b22b-30484d678dd7/download/eth-rainfall-subnat-full.csv",
)

CHIRPS_REFRESH_INTERVAL_SECONDS = int(os.environ.get("CHIRPS_REFRESH_INTERVAL_SECONDS", 24 * 60 * 60))
