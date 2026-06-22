"""
config.py — Central configuration for the Wheat Intelligence Platform.
All paths, thresholds, and constants derived from the real GEE export.
"""

import os

# ── Directory layout ──────────────────────────────────────────────────────────
ROOT_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(ROOT_DIR, "data")       # GeoTIFFs and sentinel CSV live here
OUTPUT_DIR  = os.path.join(ROOT_DIR, "outputs")    # ML model artefacts, plots, result CSVs
MODEL_DIR   = os.path.join(ROOT_DIR, "models")     # serialised model + scaler (also written to outputs/)
REPORT_DIR  = os.path.join(ROOT_DIR, "reports")    # optional PDF / HTML reports

for d in [DATA_DIR, OUTPUT_DIR, MODEL_DIR, REPORT_DIR]:
    os.makedirs(d, exist_ok=True)

# ── GEE export filenames (place these in data/) ───────────────────────────────
TIFF_FILES = {
    "ndvi":        "ndvi_wheat_ludhiana.tif",
    "ndre":        "ndre_wheat_ludhiana.tif",
    "wheat_mask":  "wheat_mask_ludhiana.tif",
    "crop_health": "crop_health_ludhiana.tif",
    "disease_risk":"disease_risk_ludhiana.tif",
    "veg_indices": "vegetation_indices_wheat_ludhiana.tif",
}

CSV_FILES = {
    "sentinel":   "sentinel_data_ludhiana.csv",
    "area_stats": "area_statistics_ludhiana_2023_24.csv",
}

# ── Spectral features expected in the sentinel CSV ────────────────────────────
# Columns confirmed present in the GEE export (system:index and .geo dropped)
RAW_BANDS    = ["B11", "B2", "B3", "B4", "B5", "B8", "B8A"]
RAW_INDICES  = ["EVI", "GNDVI", "NDRE", "NDVI", "NDWI", "SAVI"]
ENGINEERED   = ["Vegetation_Health_Score", "Stress_Score", "Red_Blue_Ratio"]

ALL_FEATURES = RAW_BANDS + RAW_INDICES + ENGINEERED   # 16 features total
TARGET_COL   = "Wheat_Class"

# ── Crop health classification thresholds ─────────────────────────────────────
# Calibrated against the crop_health_ludhiana.tif class encoding (1 / 2 / 3)
HEALTH_THRESHOLDS = {
    "healthy":  {"ndvi": 0.50, "ndre": 0.35, "evi": 0.25},   # Class 1
    "moderate": {"ndvi": 0.38, "ndre": 0.22},                  # Class 2
    # anything below → Class 3 (Poor)
}

# ── Disease risk thresholds (from area_statistics CSV) ────────────────────────
DISEASE_THRESHOLDS = {
    "high":     {"ndvi": 0.30, "ndre": 0.20},   # Class 3
    "moderate": {"ndvi": 0.50, "ndre": 0.40},   # Class 2
    # above both → low risk (Class 1)
}

# ── Known real stats from the 2023-24 GEE export ─────────────────────────────
# Used as fallbacks in the FastAPI endpoints when TIFFs are present but
# rasterio is unavailable, and as ground-truth for the Streamlit dashboard.
REAL_STATS = {
    "wheat_area_ha":       227.48,
    "avg_ndvi":            0.3791,
    "avg_ndre":            0.2622,
    "avg_gndvi":           0.3260,
    "avg_vhs":             32.81,
    "total_samples":       26600,
    "tif_shape":           (5014, 6680),
    "season":              "2023-24",
    "region":              "Ludhiana, Punjab",
    "health_distribution": {
        "healthy_ha":  9.81,
        "moderate_ha": 156.33,
        "poor_ha":     61.32,
        "healthy_pct": 4.3,
        "moderate_pct":68.7,
        "poor_pct":    27.0,
    },
    "disease_distribution": {
        "high_ha":     0.0,
        "moderate_ha": 227.42,
        "low_ha":      0.04,
    },
}

# ── ML training settings ──────────────────────────────────────────────────────
TEST_SIZE    = 0.20
RANDOM_STATE = 42
N_CLASSES    = 3     # 1=Healthy, 2=Moderate, 3=Poor

# ── FastAPI server ────────────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
