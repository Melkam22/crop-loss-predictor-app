"""Loads the trained pipeline once and serves predictions.

Reads decision_threshold / known_crops / known_region_codes from
model_card.json rather than hardcoding them, so this can't drift from the
artifact model/harvestguard_xgb.joblib was actually saved with.
"""

import json
import math
from typing import Optional

import joblib
import pandas as pd

from .config import MODEL_CARD_PATH, MODEL_PATH, REGION_SUPPORT_PATH

_pipeline = None
_model_card: Optional[dict] = None
_region_support: Optional[pd.DataFrame] = None


def load_model() -> None:
    global _pipeline, _model_card, _region_support
    _pipeline = joblib.load(MODEL_PATH)
    _model_card = json.loads(MODEL_CARD_PATH.read_text())
    _region_support = pd.read_csv(REGION_SUPPORT_PATH)


def known_regions() -> list[dict]:
    """(region_code, region_name) pairs for a frontend dropdown -- sourced from
    region_crop_support.csv (already has both columns) rather than a second,
    hand-maintained mapping."""
    return (
        _region_support[["region_code", "region_name"]]
        .drop_duplicates()
        .sort_values("region_name")
        .astype({"region_code": int})
        .to_dict(orient="records")
    )


def known_crops() -> list[str]:
    return list(_model_card["known_crops"])


def is_limited_data(region_code: int, crop_name: str) -> bool:
    match = _region_support[
        (_region_support["region_code"] == region_code) & (_region_support["crop_name"] == crop_name)
    ]
    if match.empty:
        return True  # no training rows at all for this combo -- treat as limited data
    return bool(match.iloc[0]["limited_data"])


def predict(crop_name: str, region_code: int, household_size: float, rainfall: dict) -> dict:
    if _pipeline is None or _model_card is None:
        raise RuntimeError("model not loaded -- call load_model() first")

    warnings = list(rainfall.get("_warnings", []))
    if crop_name not in _model_card["known_crops"]:
        warnings.append(f"unrecognized crop_name '{crop_name}' -- prediction ignores crop identity")
    if region_code not in _model_card["known_region_codes"]:
        warnings.append(f"unrecognized region_code {region_code} -- prediction ignores region identity")

    row = pd.DataFrame([{
        "household_size": float(household_size),
        "is_rural": 1.0,
        "rainfall_belg_mm": rainfall.get("rainfall_belg_mm", math.nan),
        "rainfall_belg_pct_of_avg": rainfall.get("rainfall_belg_pct_of_avg", math.nan),
        "rainfall_meher_mm": rainfall.get("rainfall_meher_mm", math.nan),
        "rainfall_meher_pct_of_avg": rainfall.get("rainfall_meher_pct_of_avg", math.nan),
        "crop_name": crop_name,
        "region_code": float(region_code),
    }])[_model_card["input_column_order"]]

    score = float(_pipeline.predict_proba(row)[:, 1][0])
    threshold = _model_card["decision_threshold"]

    return {
        "risk": "High" if score >= threshold else "Low",
        "score": round(score, 4),
        "score_is_probability": _model_card["score_is_probability"],
        "limited_data": is_limited_data(region_code, crop_name),
        "rainfall_used": {
            "rainfall_belg_mm": rainfall.get("rainfall_belg_mm"),
            "rainfall_belg_pct_of_avg": rainfall.get("rainfall_belg_pct_of_avg"),
            "rainfall_meher_mm": rainfall.get("rainfall_meher_mm"),
            "rainfall_meher_pct_of_avg": rainfall.get("rainfall_meher_pct_of_avg"),
        },
        "warnings": warnings,
    }
