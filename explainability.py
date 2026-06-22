"""
explainability.py — Generate feature-importance and SHAP-style plots.

Works with any tree model saved by ml_pipeline.py.
Falls back to a correlation-based SHAP approximation when the `shap`
package is not installed (pip-constrained environments).
"""

import os
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from config              import OUTPUT_DIR, DATA_DIR, CSV_FILES, ALL_FEATURES
from feature_engineering import prepare_ml_dataset

# Dark theme for all charts
plt.rcParams.update({
    "figure.facecolor": "#0f172a",
    "axes.facecolor":   "#1e293b",
    "text.color":       "white",
    "axes.labelcolor":  "white",
    "xtick.color":      "white",
    "ytick.color":      "white",
    "axes.edgecolor":   "#334155",
    "grid.color":       "#334155",
})

# ── Attempt to import shap ────────────────────────────────────────────────────
try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False
    print("[explainability] shap not installed — using correlation-based approximation.")


def _feature_importance_plot(model, feature_cols: list, save_path: str) -> None:
    """Bar chart of Gini / coefficient-based feature importances."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        print("[explainability] Model has no feature_importances_ or coef_.")
        return

    imp = pd.DataFrame({"feature": feature_cols, "importance": importances})
    imp = imp.sort_values("importance", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 7))
    colors = ["#3b82f6" if i >= len(imp) - 3 else "#334155"
              for i in range(len(imp))]
    ax.barh(imp["feature"], imp["importance"], color=colors, edgecolor="none", height=0.6)
    ax.set_xlabel("Importance Score", color="white", fontsize=11)
    ax.set_title("Feature Importance — Random Forest (Gini)",
                 color="white", fontsize=13, pad=14)
    for i, (val, feat) in enumerate(zip(imp["importance"], imp["feature"])):
        if val > 0.01:
            ax.text(val + 0.003, i, f"{val:.3f}", va="center",
                    color="white", fontsize=8.5)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"[explainability] Saved → {save_path}")


def _shap_summary_plot_approx(X_sample: pd.DataFrame,
                               importances: np.ndarray,
                               save_path: str) -> None:
    """
    SHAP-style dot plot approximation.
    Uses feature standard deviations weighted by Gini importance to
    simulate SHAP value spread.  Not mathematically equivalent to SHAP
    but visually representative for a project dashboard.
    """
    np.random.seed(42)
    features = list(X_sample.columns)
    n_features = len(features)
    n_pts = min(300, len(X_sample))

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.get_cmap("RdYlGn")

    for i, (feat, imp) in enumerate(zip(features, importances)):
        vals  = X_sample[feat].sample(n_pts, random_state=i).values
        # Simulate SHAP spread proportional to importance × std
        shap_vals = (vals - vals.mean()) / (vals.std() + 1e-8) * imp * 2.0
        jitter    = np.random.randn(n_pts) * 0.06
        # Colour encodes feature value percentile
        pct = (vals - vals.min()) / (vals.ptp() + 1e-8)
        colors_s  = cmap(pct)
        ax.scatter(shap_vals, i + jitter, c=colors_s, alpha=0.45, s=14,
                   linewidths=0)

    ax.set_yticks(range(n_features))
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel("SHAP-style impact on model output", color="white", fontsize=11)
    ax.set_title("Feature Impact Distribution (SHAP approximation)",
                 color="white", fontsize=13, pad=14)
    ax.axvline(0, color="white", linewidth=0.8, linestyle="--", alpha=0.4)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.55)
    cbar.set_label("Feature value  low → high", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"[explainability] Saved → {save_path}")


def _real_shap_plots(model, X_sample: pd.DataFrame,
                     imp_path: str, summary_path: str) -> None:
    """Generate real SHAP plots when the shap package is available."""
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)

    plt.figure(figsize=(10, 6))
    shap.plots.bar(shap_values, show=False)
    plt.tight_layout()
    plt.savefig(imp_path, dpi=120, bbox_inches="tight", facecolor="#0f172a")
    plt.close()

    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(summary_path, dpi=120, bbox_inches="tight", facecolor="#0f172a")
    plt.close()
    print(f"[explainability] Real SHAP plots saved.")


def generate_shap_plots(
        model_path:  str = None,
        scaler_path: str = None,
        data_path:   str = None,
        n_sample:    int = 300,
) -> None:
    """
    Main entry point.  Loads the saved model + scaler, samples the dataset,
    and writes shap_importance.png and shap_summary.png to OUTPUT_DIR.
    """
    model_path  = model_path  or os.path.join(OUTPUT_DIR, "best_model.pkl")
    scaler_path = scaler_path or os.path.join(OUTPUT_DIR, "scaler.pkl")
    data_path   = data_path   or os.path.join(DATA_DIR, CSV_FILES["sentinel"])

    # Check prerequisites
    for p in [model_path, scaler_path]:
        if not os.path.exists(p):
            print(f"[explainability] Missing: {p}")
            print("  → Run ml_pipeline.py first to generate model artefacts.")
            return

    model  = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    X, _ = prepare_ml_dataset(data_path)
    X_sample = X.sample(min(n_sample, len(X)), random_state=42)
    X_scaled = pd.DataFrame(
        scaler.transform(X_sample), columns=X_sample.columns
    )

    imp_path     = os.path.join(OUTPUT_DIR, "shap_importance.png")
    summary_path = os.path.join(OUTPUT_DIR, "shap_summary.png")

    if _HAS_SHAP and hasattr(model, "estimators_"):
        # Real SHAP (requires shap package + tree model)
        _real_shap_plots(model, X_scaled, imp_path, summary_path)
    else:
        # Fallback: Gini importance bar chart + correlation-based dot plot
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        else:
            importances = np.ones(len(ALL_FEATURES)) / len(ALL_FEATURES)

        _feature_importance_plot(model, list(X_scaled.columns), imp_path)
        _shap_summary_plot_approx(X_scaled, importances, summary_path)

    # Also generate NDVI and crop-health preview images
    _generate_raster_previews()


def _generate_raster_previews() -> None:
    """
    Save colourised preview PNGs of NDVI and crop_health TIFFs.
    These are served by Streamlit and displayed in the FastAPI /outputs route.
    """
    import matplotlib.colors as mcolors

    # NDVI preview
    ndvi_tif = os.path.join(DATA_DIR, "ndvi_wheat_ludhiana.tif")
    if os.path.exists(ndvi_tif):
        try:
            from PIL import Image as PILImage
            ndvi_arr  = np.array(PILImage.open(ndvi_tif), dtype=np.float32)
            step      = 10
            ndvi_small = ndvi_arr[::step, ::step]
            fig, ax   = plt.subplots(figsize=(9, 7))
            im = ax.imshow(ndvi_small, cmap="RdYlGn", vmin=0.30, vmax=0.65)
            plt.colorbar(im, ax=ax, label="NDVI", shrink=0.75)
            ax.set_title("NDVI Map — Ludhiana Wheat 2023-24",
                         color="white", fontsize=12)
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "ndvi_preview.png"),
                        dpi=110, bbox_inches="tight", facecolor="#0f172a")
            plt.close()
            print("[explainability] Saved → outputs/ndvi_preview.png")
        except Exception as e:
            print(f"[explainability] NDVI preview skipped: {e}")

    # Crop-health class map
    ch_tif = os.path.join(DATA_DIR, "crop_health_ludhiana.tif")
    if os.path.exists(ch_tif):
        try:
            ch_arr  = np.array(PILImage.open(ch_tif), dtype=np.float32)
            step    = 10
            ch_small = ch_arr[::step, ::step]
            cmap_h  = mcolors.ListedColormap(["#22c55e", "#eab308", "#ef4444"])
            norm    = mcolors.BoundaryNorm([0.5, 1.5, 2.5, 3.5], 3)
            fig, ax = plt.subplots(figsize=(9, 7))
            ax.imshow(ch_small, cmap=cmap_h, norm=norm)
            patches = [
                mpatches.Patch(color="#22c55e", label="Healthy (1)"),
                mpatches.Patch(color="#eab308", label="Moderate (2)"),
                mpatches.Patch(color="#ef4444", label="Poor (3)"),
            ]
            ax.legend(handles=patches, loc="lower right",
                      facecolor="#1e293b", edgecolor="#334155",
                      labelcolor="white", fontsize=9)
            ax.set_title("Crop Health Classification Map",
                         color="white", fontsize=12)
            ax.axis("off")
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_DIR, "crop_health_preview.png"),
                        dpi=110, bbox_inches="tight", facecolor="#0f172a")
            plt.close()
            print("[explainability] Saved → outputs/crop_health_preview.png")
        except Exception as e:
            print(f"[explainability] Crop-health preview skipped: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    generate_shap_plots()
