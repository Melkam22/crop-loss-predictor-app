"""Live rainfall features for inference.

Ports the exact transformation from notebooks/01_data_exploration.ipynb Part 8
(raw CHIRPS -> region-level dekadal rows) and Part 9 (season_agg) so live
predictions use the identical formula the model was trained on. The only new
piece is picking *which* year's Belg/Meher season to use, since training only
ever saw complete seasons but a live request can land mid-season.
"""

import math
from datetime import date
from typing import Optional

import pandas as pd
import requests

from .config import CHIRPS_CSV_PATH, CHIRPS_RESOURCE_URL

SEASON_MONTHS = {
    "belg": range(2, 6),    # Feb-May
    "meher": range(6, 10),  # Jun-Sep
}
SEASON_END_MONTH = {"belg": 5, "meher": 9}
SEASON_DEKAD_COUNT = {label: len(months) * 3 for label, months in SEASON_MONTHS.items()}  # 12 each

# Verbatim from 01_data_exploration.ipynb Part 8 (cell 47).
RAINFALL_RENAME = {
    "date": "date",
    "PCODE": "admin_pcode",
    "rfh": "rainfall_dekad_mm",
    "rfh_avg": "rainfall_dekad_avg_mm",
    "r1h": "rainfall_1month_mm",
    "r1h_avg": "rainfall_1month_avg_mm",
    "r3h": "rainfall_3month_mm",
    "r3h_avg": "rainfall_3month_avg_mm",
    "rfq": "rainfall_dekad_pct_of_avg",
    "r1q": "rainfall_1month_pct_of_avg",
    "r3q": "rainfall_3month_pct_of_avg",
    "version": "data_version",
}

_dekadal_cache: Optional[pd.DataFrame] = None


def refresh_chirps_csv() -> None:
    """Download the current HDX resource and overwrite the local raw CSV in place."""
    resp = requests.get(CHIRPS_RESOURCE_URL, timeout=120)
    resp.raise_for_status()
    CHIRPS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    CHIRPS_CSV_PATH.write_bytes(resp.content)
    load_dekadal(force=True)


def load_dekadal(force: bool = False) -> pd.DataFrame:
    """Parse the raw CHIRPS CSV into region-level dekadal rows (Part 8's logic).
    Cached in memory; pass force=True to reparse after a refresh."""
    global _dekadal_cache
    if _dekadal_cache is not None and not force:
        return _dekadal_cache

    rain_raw = pd.read_csv(CHIRPS_CSV_PATH, low_memory=False)
    rainfall_region = (
        rain_raw[rain_raw["adm_level"] == 1]
        .query("version == 'final'")
        [list(RAINFALL_RENAME)]
        .rename(columns=RAINFALL_RENAME)
        .copy()
    )
    rainfall_region["region_code"] = (
        rainfall_region["admin_pcode"].str.replace("ET", "", regex=False).astype(int)
    )
    rainfall_region["date"] = pd.to_datetime(rainfall_region["date"])
    rainfall_region["year"] = rainfall_region["date"].dt.year
    rainfall_region["month"] = rainfall_region["date"].dt.month

    _dekadal_cache = rainfall_region
    return rainfall_region


def _season_agg_for(df: pd.DataFrame, region_code: int, year: int, label: str) -> Optional[dict]:
    months = SEASON_MONTHS[label]
    sub = df[(df["region_code"] == region_code) & (df["year"] == year) & (df["month"].isin(months))]
    if sub.empty:
        return None
    total_mm = float(sub["rainfall_dekad_mm"].sum())
    avg_mm = float(sub["rainfall_dekad_avg_mm"].sum())
    pct_of_avg = round(total_mm / avg_mm * 100, 1) if avg_mm else None
    return {"mm": round(total_mm, 1), "pct_of_avg": pct_of_avg, "n_dekads": len(sub)}


def _most_recent_complete_year(today: date, label: str) -> int:
    end_month = SEASON_END_MONTH[label]
    return today.year if today.month > end_month else today.year - 1


def season_features(region_code: int, today: Optional[date] = None) -> dict:
    """rainfall_{belg,meher}_{mm,pct_of_avg} for region_code, using the most
    recently COMPLETE season of each type -- the model was only ever trained on
    complete-season totals. Falls back further if CHIRPS hasn't finalized
    ('final' vs 'prelim') all of a season's dekads yet."""
    today = today or date.today()
    df = load_dekadal()
    out: dict = {}
    warnings: list[str] = []

    for label in ("belg", "meher"):
        year = _most_recent_complete_year(today, label)
        expected = SEASON_DEKAD_COUNT[label]
        result = _season_agg_for(df, region_code, year, label)
        tries = 0
        while (result is None or result["n_dekads"] < expected) and tries < 3:
            year -= 1
            result = _season_agg_for(df, region_code, year, label)
            tries += 1

        if result is None:
            warnings.append(f"no {label} rainfall data found for region_code {region_code}")
            out[f"rainfall_{label}_mm"] = math.nan
            out[f"rainfall_{label}_pct_of_avg"] = math.nan
            continue

        if result["n_dekads"] < expected:
            warnings.append(
                f"{label} rainfall for region_code {region_code} ({year}) is based on "
                f"partial data ({result['n_dekads']}/{expected} dekads)"
            )
        out[f"rainfall_{label}_mm"] = result["mm"]
        out[f"rainfall_{label}_pct_of_avg"] = result["pct_of_avg"] if result["pct_of_avg"] is not None else math.nan
        out[f"_{label}_year_used"] = year

    out["_warnings"] = warnings
    return out
