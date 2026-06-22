"""
app_driver.py — CLI driver to run the full backend pipeline end-to-end.

Usage:
    python app_driver.py              # full pipeline
    python app_driver.py --ml-only   # skip explainability
    python app_driver.py --xai-only  # skip ML training (uses existing model)

Requires the GEE data files in data/:
    sentinel_data_ludhiana.csv
    ndvi_wheat_ludhiana.tif
    crop_health_ludhiana.tif
    disease_risk_ludhiana.tif
    wheat_mask_ludhiana.tif
"""

import sys
import os

from config              import DATA_DIR, OUTPUT_DIR, CSV_FILES
from feature_engineering import prepare_ml_dataset     # FIXED: was prepare_dataset
from ml_pipeline         import train_and_serialize_models
from explainability      import generate_shap_plots


def run_ml(csv_path: str) -> str:
    """Run feature engineering + model training. Returns best model name."""
    print("\n" + "="*60)
    print("STEP 1 — Feature Engineering")
    print("="*60)
    X, y = prepare_ml_dataset(csv_path)

    print("\n" + "="*60)
    print("STEP 2 — Model Training & Evaluation")
    print("="*60)
    best_name, metrics_df = train_and_serialize_models(X, y)

    print(f"\n✅ Best model: {best_name}")
    print(metrics_df.to_string(index=False))
    return best_name


def run_xai(csv_path: str) -> None:
    """Generate explainability plots using the saved model."""
    print("\n" + "="*60)
    print("STEP 3 — Explainability (Feature Importance + SHAP)")
    print("="*60)
    generate_shap_plots(data_path=csv_path)
    print("✅ Plots saved to outputs/")


def main():
    csv_path = os.path.join(DATA_DIR, CSV_FILES["sentinel"])

    if not os.path.exists(csv_path):
        print(f"\n❌ Missing: {csv_path}")
        print("  Place sentinel_data_ludhiana.csv in the data/ folder and re-run.")
        sys.exit(1)

    ml_only  = "--ml-only"  in sys.argv
    xai_only = "--xai-only" in sys.argv

    if xai_only:
        run_xai(csv_path)
    elif ml_only:
        run_ml(csv_path)
    else:
        # Full pipeline
        run_ml(csv_path)
        run_xai(csv_path)

    print("\n" + "="*60)
    print("Pipeline complete. Start the API with:  python main.py")
    print("Start the dashboard with:              streamlit run app.py")
    print("="*60)


if __name__ == "__main__":
    main()
