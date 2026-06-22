"""
utils.py — Shared utilities for raster I/O, statistics, and data helpers.

Works with PIL/tifffile when rasterio is unavailable (e.g. pip-constrained
environments), and upgrades transparently to rasterio when it is installed.
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple

# ── Raster back-end detection ─────────────────────────────────────────────────
try:
    import rasterio
    _HAS_RASTERIO = True
except ImportError:
    _HAS_RASTERIO = False

try:
    from PIL import Image
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False


def load_raster(file_path: str) -> Optional[np.ndarray]:
    """
    Load a single-band GeoTIFF as a float32 NumPy array.
    Returns None and prints a warning if the file is missing.
    NaN pixels are preserved; no nodata substitution is applied here.
    """
    if not os.path.exists(file_path):
        print(f"[utils] Warning: {file_path} not found.")
        return None

    if _HAS_RASTERIO:
        with rasterio.open(file_path) as src:
            data = src.read(1).astype(np.float32)
            # Convert rasterio nodata to NaN
            if src.nodata is not None:
                data[data == src.nodata] = np.nan
            return data

    if _HAS_PIL:
        img = Image.open(file_path)
        arr = np.array(img, dtype=np.float32)
        # GEE exports occasionally write 0 for masked areas on float TIFFs;
        # we leave that to the caller to decide how to filter.
        return arr

    raise RuntimeError(
        "Neither rasterio nor PIL is available. "
        "Install one with: pip install Pillow"
    )


def get_raster_stats(data: np.ndarray) -> Dict[str, float]:
    """
    Compute basic statistics over valid (non-NaN, non-masked) pixels.
    Handles both masked arrays and plain ndarrays.
    """
    if data is None:
        return {}

    if isinstance(data, np.ma.MaskedArray):
        valid = data.compressed().astype(np.float64)
    else:
        valid = data[~np.isnan(data)].astype(np.float64)

    if valid.size == 0:
        return {"mean": 0.0, "min": 0.0, "max": 0.0, "std": 0.0, "count": 0}

    return {
        "mean":  float(np.mean(valid)),
        "min":   float(np.min(valid)),
        "max":   float(np.max(valid)),
        "std":   float(np.std(valid)),
        "count": int(valid.size),
    }


def raster_area_ha(data: np.ndarray,
                   pixel_size_m: float = 10.0,
                   mask_value: float = 1.0) -> float:
    """
    Calculate the area in hectares occupied by pixels equal to mask_value.
    Default pixel size is 10 m (Sentinel-2).
    """
    if data is None:
        return 0.0
    pixel_area_ha = (pixel_size_m ** 2) / 10_000
    return float(np.sum(data == mask_value)) * pixel_area_ha


def safe_divide(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise division that returns 0 where denominator is 0."""
    return np.divide(a, b, out=np.zeros_like(a, dtype=np.float32), where=(b != 0))


def load_sentinel_csv(csv_path: str) -> pd.DataFrame:
    """
    Load the GEE-exported sentinel CSV, drop metadata columns,
    and return a clean DataFrame ready for feature engineering.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Sentinel CSV not found at '{csv_path}'.\n"
            "Expected: data/sentinel_data_ludhiana.csv (GEE export)."
        )
    df = pd.read_csv(csv_path)
    # Drop GEE metadata columns that are not spectral features
    drop_cols = [c for c in ["system:index", ".geo"] if c in df.columns]
    df = df.drop(columns=drop_cols)
    return df


def load_area_stats(csv_path: str) -> Dict[str, float]:
    """
    Load the GEE area statistics CSV and return a {metric: value} dict.
    """
    if not os.path.exists(csv_path):
        return {}
    df = pd.read_csv(csv_path)
    if "metric" in df.columns and "value" in df.columns:
        return dict(zip(df["metric"], df["value"]))
    return {}


def get_tiff_stats_from_path(filepath: str) -> Optional[Dict]:
    """
    High-level helper: open a TIF, compute stats, and return them
    plus the area in hectares (for the wheat_mask use-case).
    """
    data = load_raster(filepath)
    if data is None:
        return None
    stats = get_raster_stats(data)
    stats["area_ha"] = raster_area_ha(data, mask_value=1.0)
    return stats


def colorise_ndvi(value: float) -> str:
    """Return a hex colour string for an NDVI value (for Streamlit display)."""
    if value > 0.55:  return "#006400"   # dark green — high vigor
    if value > 0.45:  return "#22c55e"   # green — moderate vigor
    if value > 0.40:  return "#86efac"   # light green — low vigor
    if value > 0.38:  return "#fde68a"   # yellow — marginal
    return "#f87171"                      # red — stressed


def health_class_label(cls: int) -> str:
    return {1: "Healthy", 2: "Moderate", 3: "Poor"}.get(cls, "Unknown")


def risk_class_label(cls: int) -> str:
    return {1: "Low Risk", 2: "Moderate Risk", 3: "High Risk"}.get(cls, "Unknown")
