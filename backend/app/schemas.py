from typing import Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    crop_name: str = Field(..., description="Crop name, e.g. 'MAIZE'. See model_card.json's known_crops.")
    region_code: int = Field(..., description="LSMS region code, e.g. 1 for Tigray.")
    household_size: float = Field(..., gt=0, description="Number of people in the household.")


class RainfallUsed(BaseModel):
    rainfall_belg_mm: Optional[float]
    rainfall_belg_pct_of_avg: Optional[float]
    rainfall_meher_mm: Optional[float]
    rainfall_meher_pct_of_avg: Optional[float]


class PredictResponse(BaseModel):
    risk: str
    score: float
    score_is_probability: bool
    limited_data: bool
    rainfall_used: RainfallUsed
    warnings: list[str]


class RegionOption(BaseModel):
    region_code: int
    region_name: str
