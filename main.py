"""
main.py — FastAPI backend for the Wheat Intelligence Platform.

Endpoints
---------
GET  /api/health          — liveness probe
GET  /api/stats           — area / index statistics (from TIFFs or baked-in fallback)
GET  /api/ml-results      — model benchmark table
GET  /api/feature-importance — feature importance from RF
POST /api/search          — point-based local NDVI query
GET  /api/area-stats      — raw area_statistics CSV as JSON
GET  /api/outputs/{name}  — serve generated PNG plots
"""

import os
import uvicorn
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors    import CORSMiddleware
from fastapi.staticfiles        import StaticFiles
from fastapi.responses          import FileResponse
from pydantic                   import BaseModel, model_validator
import joblib

try:
    from wheat_ludhiana_app.config import (
        DATA_DIR, OUTPUT_DIR, CSV_FILES, TIFF_FILES, REAL_STATS, API_HOST, API_PORT
    )
    from wheat_ludhiana_app.utils import load_raster, get_raster_stats, load_area_stats
except Exception:
    from config import (
        DATA_DIR, OUTPUT_DIR, CSV_FILES, TIFF_FILES, REAL_STATS, API_HOST, API_PORT
    )
    from utils import load_raster, get_raster_stats, load_area_stats

# Optional geocoding
try:
    from geopy.geocoders import Nominatim
except Exception:
    Nominatim = None

# ── App setup ─────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Wheat Intelligence API",
    description="Sentinel-2 / GEE satellite analytics for Ludhiana wheat 2023-24",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"],  # Vite dev server
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Serve output plots (SHAP, NDVI preview, etc.)
if os.path.exists(OUTPUT_DIR):
    app.mount("/outputs", StaticFiles(directory=OUTPUT_DIR), name="outputs")

# Serve raw TIFFs for geo-raster (Leaflet)
if os.path.exists(DATA_DIR):
    app.mount("/api/raster", StaticFiles(directory=DATA_DIR), name="raster")


# ── Request / response models ─────────────────────────────────────────────────
class SearchRequest(BaseModel):
    # Either provide latitude+longitude, or a place-name `query` string
    latitude:  float | None = None
    longitude: float | None = None
    query:     str | None = None

    @model_validator(mode="after")
    def must_have_coords_or_query(self):
        if self.query is None and (self.latitude is None or self.longitude is None):
            raise ValueError("Provide either 'query' or both 'latitude' and 'longitude'.")
        return self


# ── Helper: read TIF stats with PIL fallback ──────────────────────────────────
def _tif_stats(filename: str):
    path = os.path.join(DATA_DIR, filename)
    if not os.path.exists(path):
        return None
    arr = load_raster(path)
    if arr is None:
        return None
    return get_raster_stats(arr)


def _tif_array(filename: str):
    path = os.path.join(DATA_DIR, filename)
    return load_raster(path)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "2.0.0", "season": REAL_STATS["season"]}


@app.get("/api/stats")
async def get_stats():
    """
    Return area statistics and index means.
    Tries to read live TIF files; falls back to the baked-in GEE values
    from config.py when the data/ folder is absent.
    """
    # Try live TIFFs
    ndvi_stats = _tif_stats(TIFF_FILES["ndvi"])
    ndre_stats = _tif_stats(TIFF_FILES["ndre"])
    wm_arr     = _tif_array(TIFF_FILES["wheat_mask"])

    # Load area stats CSV
    area_csv  = os.path.join(DATA_DIR, CSV_FILES["area_stats"])
    area_dict = load_area_stats(area_csv)

    wheat_area = (
        area_dict.get("Total_Wheat_Area_ha")
        or (float(np.sum(wm_arr == 1.0)) * 0.01 if wm_arr is not None else None)
        or REAL_STATS["wheat_area_ha"]
    )

    avg_ndvi = (
        ndvi_stats["mean"] if ndvi_stats else REAL_STATS["avg_ndvi"]
    )
    avg_ndre = (
        ndre_stats["mean"] if ndre_stats else REAL_STATS["avg_ndre"]
    )

    hd = REAL_STATS["health_distribution"]
    return {
        "wheat_area": round(float(wheat_area), 2),
        "avg_ndvi":   round(float(avg_ndvi),   4),
        "avg_ndre":   round(float(avg_ndre),   4),
        "season":     REAL_STATS["season"],
        "region":     REAL_STATS["region"],
        "health_distribution": {
            "healthy":  hd["healthy_pct"],
            "moderate": hd["moderate_pct"],
            "stressed": hd["poor_pct"],
        },
    }


@app.get("/api/ml-results")
async def get_ml_results():
    """Return model benchmark table. Reads CSV if available, else returns baked-in values."""
    results_path = os.path.join(OUTPUT_DIR, "model_comparison_results.csv")
    if os.path.exists(results_path):
        df = pd.read_csv(results_path)
        return df.to_dict(orient="records")

    # Baked-in results from the training run on the real GEE data
    return [
        {"Model": "RandomForest",       "Accuracy": 0.9998, "F1_Score": 0.9998, "ROC_AUC": 1.0000},
        {"Model": "GradientBoosting",   "Accuracy": 0.9994, "F1_Score": 0.9995, "ROC_AUC": 0.9991},
        {"Model": "LogisticRegression", "Accuracy": 0.8923, "F1_Score": 0.8902, "ROC_AUC": 0.9715},
    ]


@app.get("/api/feature-importance")
async def get_feature_importance():
    """Return feature importance from the saved Random Forest model."""
    imp_path = os.path.join(OUTPUT_DIR, "feature_importance.csv")
    if os.path.exists(imp_path):
        df = pd.read_csv(imp_path)
        return df.to_dict(orient="records")

    # Baked-in from the actual RF training run
    return [
        {"feature": "NDVI",                  "importance": 0.5231},
        {"feature": "NDRE",                  "importance": 0.1870},
        {"feature": "Vegetation_Health_Score","importance": 0.1550},
        {"feature": "B4",                    "importance": 0.0318},
        {"feature": "SAVI",                  "importance": 0.0192},
        {"feature": "B2",                    "importance": 0.0167},
        {"feature": "Red_Blue_Ratio",        "importance": 0.0107},
        {"feature": "EVI",                   "importance": 0.0105},
    ]


@app.get("/api/area-stats")
async def get_area_stats():
    """Return the raw area_statistics CSV as JSON."""
    area_csv = os.path.join(DATA_DIR, CSV_FILES["area_stats"])
    if os.path.exists(area_csv):
        df = pd.read_csv(area_csv)
        df = df.drop(columns=["system:index", ".geo"], errors="ignore")
        return df.to_dict(orient="records")

    hd = REAL_STATS["health_distribution"]
    dd = REAL_STATS["disease_distribution"]
    return [
        {"metric": "Total_Wheat_Area_ha",     "value": REAL_STATS["wheat_area_ha"], "season": "2023-24"},
        {"metric": "Healthy_Wheat_ha",         "value": hd["healthy_ha"],            "season": "2023-24"},
        {"metric": "Moderate_Wheat_ha",        "value": hd["moderate_ha"],           "season": "2023-24"},
        {"metric": "Poor_Health_Wheat_ha",     "value": hd["poor_ha"],               "season": "2023-24"},
        {"metric": "High_Disease_Risk_ha",     "value": dd["high_ha"],               "season": "2023-24"},
        {"metric": "Moderate_Disease_Risk_ha", "value": dd["moderate_ha"],           "season": "2023-24"},
        {"metric": "Low_Disease_Risk_ha",      "value": dd["low_ha"],                "season": "2023-24"},
    ]


@app.post("/api/search")
async def spatial_search(req: SearchRequest):
    """
    Query local NDVI/NDRE for a lat/lon point.
    Requires rasterio for geospatial indexing.  Falls back to a jittered
    value derived from the dataset mean when rasterio is unavailable.
    """
    # Resolve query -> coordinates when needed
    lat = req.latitude
    lon = req.longitude

    if req.query:
        if Nominatim is None:
            raise HTTPException(status_code=500, detail="Geocoding not available (missing geopy).")
        geolocator = Nominatim(user_agent="wheat-intel-app")
        try:
            loc = geolocator.geocode(req.query, exactly_one=True, timeout=10)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Geocoding error: {e}")
        if not loc:
            raise HTTPException(status_code=404, detail=f"Location not found: '{req.query}'")
        lat, lon = loc.latitude, loc.longitude

    # Core: attempt to read rasters using rasterio; if unavailable or missing files, fall back
    try:
        import rasterio
        ndvi_path = os.path.join(DATA_DIR, TIFF_FILES["ndvi"])
        ndre_path = os.path.join(DATA_DIR, TIFF_FILES["ndre"])

        if not os.path.exists(ndvi_path):
            raise FileNotFoundError("NDVI TIF missing")

        with rasterio.open(ndvi_path) as src:
            try:
                py, px = src.index(lon, lat)
            except Exception:
                raise HTTPException(status_code=400, detail="Location is outside the current study area.")
            win    = rasterio.windows.Window(px - 2, py - 2, 5, 5)
            data   = src.read(1, window=win).astype(np.float32)
            local_ndvi = np.nanmean(data)
            if np.isnan(local_ndvi):
                local_ndvi = REAL_STATS["avg_ndvi"]

        local_ndre = REAL_STATS["avg_ndre"]
        if os.path.exists(ndre_path):
            with rasterio.open(ndre_path) as src:
                py2, px2 = src.index(lon, lat)
                win2     = rasterio.windows.Window(px2 - 2, py2 - 2, 5, 5)
                data2    = src.read(1, window=win2).astype(np.float32)
                local_ndre = np.nanmean(data2)
                if np.isnan(local_ndre):
                    local_ndre = REAL_STATS["avg_ndre"]

        return {
            "latitude":   round(lat, 6),
            "longitude":  round(lon, 6),
            "local_ndvi": round(float(local_ndvi), 4),
            "local_ndre": round(float(local_ndre), 4),
            "source":     "rasterio_live",
        }

    except FileNotFoundError:
        # TIFF data is not available locally; fall back to baked-in stats.
        rng        = np.random.default_rng(seed=int(abs(lat * 1000)))
        jitter     = 1 + (rng.random() * 0.12 - 0.06)
        local_ndvi = round(REAL_STATS["avg_ndvi"] * jitter, 4)
        local_ndre = round(REAL_STATS["avg_ndre"] * jitter, 4)
        return {
            "latitude":   round(lat, 6),
            "longitude":  round(lon, 6),
            "local_ndvi": local_ndvi,
            "local_ndre": local_ndre,
            "source":     "fallback_from_dataset_mean",
        }
    except ImportError:
        # Fallback: jitter around dataset mean (still informative for demo)
        rng        = np.random.default_rng(seed=int(abs(lat * 1000)))
        jitter     = 1 + (rng.random() * 0.12 - 0.06)
        local_ndvi = round(REAL_STATS["avg_ndvi"] * jitter, 4)
        local_ndre = round(REAL_STATS["avg_ndre"] * jitter, 4)
        return {
            "latitude":   round(lat, 6),
            "longitude":  round(lon, 6),
            "local_ndvi": local_ndvi,
            "local_ndre": local_ndre,
            "source":     "fallback_from_dataset_mean",
        }
    except HTTPException:
        raise
    except Exception:
        # Generic fallback if unexpected error occurs
        rng        = np.random.default_rng(seed=int(abs(lat * 1000)))
        jitter     = 1 + (rng.random() * 0.12 - 0.06)
        local_ndvi = round(REAL_STATS["avg_ndvi"] * jitter, 4)
        local_ndre = round(REAL_STATS["avg_ndre"] * jitter, 4)
        return {
            "latitude":   round(lat, 6),
            "longitude":  round(lon, 6),
            "local_ndvi": local_ndvi,
            "local_ndre": local_ndre,
            "source":     "fallback_from_dataset_mean",
        }


# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host=API_HOST, port=API_PORT, reload=True)
