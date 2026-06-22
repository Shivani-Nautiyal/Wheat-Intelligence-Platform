"""
feature_engineering.py — Spectral feature engineering for the wheat ML pipeline.

Reads the GEE-exported sentinel_data_ludhiana.csv, computes three agronomic
derived features, synthesises the Wheat_Class target from real NDVI/NDRE/EVI
thresholds, and returns X, y ready for sklearn.
"""

import pandas as pd
import numpy as np
import os
from config import (
    ALL_FEATURES, TARGET_COL, HEALTH_THRESHOLDS, CSV_FILES, DATA_DIR
)
from utils import load_sentinel_csv


# ── Feature engineering ───────────────────────────────────────────────────────

def calculate_advanced_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute three agronomic derived features from raw Sentinel-2 bands.

    1. Vegetation_Health_Score
       Weighted blend of the three chlorophyll-sensitive indices.
       Scale 0–100: NDVI×40 % + NDRE×30 % + GNDVI×30 %

    2. Stress_Score
       NIR / SWIR ratio — inverse proxy for canopy water / moisture stress.
       High value → healthy water status; low value → drought stress.

    3. Red_Blue_Ratio
       Red (B4) / Blue (B2) ratio — sensitive to anthocyanin accumulation
       and early senescence signals.

    Returns a copy of df with the three new columns appended.
    """
    df = df.copy()

    # 1. Vegetation Health Score (0–100 scale)
    df["Vegetation_Health_Score"] = (
        (df["NDVI"]  * 0.40) +
        (df["NDRE"]  * 0.30) +
        (df["GNDVI"] * 0.30)
    ) * 100

    # 2. Stress Score — NIR / SWIR  (avoid division by zero)
    df["Stress_Score"] = df["B8"] / (df["B11"] + 1e-6)

    # 3. Red-Blue Ratio
    df["Red_Blue_Ratio"] = np.where(
        df["B2"] != 0,
        df["B4"] / df["B2"],
        0.0
    )

    return df


def synthesise_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Derive Wheat_Class (1 / 2 / 3) from spectral indices.

    Thresholds are calibrated to match the pixel classification in
    crop_health_ludhiana.tif (exported from GEE):
      Class 1 — Healthy   : NDVI > 0.50, NDRE > 0.35, EVI > 0.25
      Class 2 — Moderate  : NDVI > 0.38, NDRE > 0.22
      Class 3 — Poor      : everything else

    Result distribution on the 2023-24 dataset:
      Class 1 → ~19 pixels (0.07 %)
      Class 2 → ~6 973 pixels (26.2 %)
      Class 3 → ~19 608 pixels (73.7 %)
    """
    h  = HEALTH_THRESHOLDS["healthy"]
    m  = HEALTH_THRESHOLDS["moderate"]

    conditions = [
        (df["NDVI"] > h["ndvi"]) & (df["NDRE"] > h["ndre"]) & (df["EVI"] > h["evi"]),
        (df["NDVI"] > m["ndvi"]) & (df["NDRE"] > m["ndre"]),
    ]
    choices = [1, 2]
    df[TARGET_COL] = np.select(conditions, choices, default=3)
    return df


def prepare_ml_dataset(csv_path: str):
    """
    Full pipeline: load CSV → feature engineering → target synthesis → X, y.

    Parameters
    ----------
    csv_path : str
        Path to sentinel_data_ludhiana.csv (GEE export).

    Returns
    -------
    X : pd.DataFrame  — shape (N, 16), all numeric features
    y : pd.Series     — integer class labels 1 / 2 / 3
    """
    df = load_sentinel_csv(csv_path)
    df = calculate_advanced_metrics(df)
    df = synthesise_target(df)

    # Validate all expected features are present
    missing = [f for f in ALL_FEATURES if f not in df.columns]
    if missing:
        raise ValueError(
            f"Missing features after engineering: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    X = df[ALL_FEATURES].astype(np.float32)
    y = df[TARGET_COL].astype(int)

    print(f"[feature_engineering] Dataset ready: {X.shape[0]} samples × {X.shape[1]} features")
    print(f"[feature_engineering] Class distribution:\n{y.value_counts().sort_index().to_string()}")
    return X, y


# ── Standalone test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    csv_path = os.path.join(DATA_DIR, CSV_FILES["sentinel"])
    try:
        X, y = prepare_ml_dataset(csv_path)
        print("\nSample features (first 3 rows):")
        print(X.head(3).to_string())
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
