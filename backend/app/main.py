import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from . import model_service, rainfall_service
from .config import CHIRPS_CSV_PATH, CHIRPS_REFRESH_INTERVAL_SECONDS
from .schemas import PredictRequest, PredictResponse, RegionOption

logger = logging.getLogger("harvestguard")


async def _refresh_loop() -> None:
    while True:
        await asyncio.sleep(CHIRPS_REFRESH_INTERVAL_SECONDS)
        try:
            await asyncio.to_thread(rainfall_service.refresh_chirps_csv)
            logger.info("CHIRPS refresh complete")
        except Exception:
            logger.exception("CHIRPS refresh failed; keeping previous data")


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service.load_model()
    if CHIRPS_CSV_PATH.exists():
        rainfall_service.load_dekadal()
    else:
        await asyncio.to_thread(rainfall_service.refresh_chirps_csv)
    task = asyncio.create_task(_refresh_loop())
    yield
    task.cancel()


app = FastAPI(title="HarvestGuard API", lifespan=lifespan)

# Permissive for now (dev/handoff stage) -- tighten allow_origins once the
# frontend's real domain is known.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    rainfall = rainfall_service.season_features(req.region_code)
    return model_service.predict(req.crop_name, req.region_code, req.household_size, rainfall)


@app.get("/regions", response_model=list[RegionOption])
def regions():
    """(region_code, region_name) pairs for a frontend dropdown."""
    return model_service.known_regions()


@app.get("/crops", response_model=list[str])
def crops():
    """Valid crop_name values for a frontend dropdown."""
    return model_service.known_crops()


@app.post("/admin/refresh-rainfall")
async def refresh_rainfall():
    try:
        await asyncio.to_thread(rainfall_service.refresh_chirps_csv)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"CHIRPS refresh failed: {exc}")
    return {"status": "refreshed"}
