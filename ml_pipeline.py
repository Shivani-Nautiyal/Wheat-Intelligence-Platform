"""
ml_pipeline.py — Train, evaluate, and serialise wheat-health classifiers.

Models used (sklearn-only; no xgboost/catboost dependency required):
  • RandomForestClassifier      — best model on this dataset
  • GradientBoostingClassifier  — strong baseline
  • LogisticRegression          — linear baseline

Outputs written to outputs/ :
  best_model.pkl                — serialised best model
  scaler.pkl                    — fitted StandardScaler
  feature_cols.pkl              — ordered feature column list
  model_comparison_results.csv  — accuracy / F1 / AUC per model
  feature_importance.csv        — RF Gini importances
"""

import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import StandardScaler
from sklearn.metrics         import (accuracy_score, f1_score,
                                     roc_auc_score, classification_report)
from sklearn.ensemble        import (RandomForestClassifier,
                                     GradientBoostingClassifier)
from sklearn.linear_model    import LogisticRegression

from config              import (ALL_FEATURES, TARGET_COL, OUTPUT_DIR,
                                  DATA_DIR, CSV_FILES, TEST_SIZE, RANDOM_STATE)
from feature_engineering import prepare_ml_dataset


# ── Model registry ────────────────────────────────────────────────────────────
def _build_models():
    return {
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=1,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=5,
            random_state=RANDOM_STATE,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            solver="lbfgs",
            random_state=RANDOM_STATE,
        ),
    }


# ── Training ──────────────────────────────────────────────────────────────────
def train_and_serialize_models(X: pd.DataFrame, y: pd.Series):
    """
    Train all models, compute metrics, save artefacts, return summary.

    Returns
    -------
    best_name   : str            — name of the winning model
    metrics_df  : pd.DataFrame  — per-model metrics table
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Scale
    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_s  = pd.DataFrame(scaler.transform(X_test),      columns=X.columns)

    models   = _build_models()
    results  = []
    best_f1, best_model_obj, best_name = 0.0, None, ""

    for name, model in models.items():
        print(f"[ml_pipeline] Training {name} …")
        model.fit(X_train_s, y_train)

        y_pred = model.predict(X_test_s)
        y_prob = model.predict_proba(X_test_s)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred, average="weighted")
        auc = roc_auc_score(y_test, y_prob, multi_class="ovr")

        results.append({
            "Model":    name,
            "Accuracy": round(acc, 4),
            "F1_Score": round(f1,  4),
            "ROC_AUC":  round(auc, 4),
        })
        print(f"  Accuracy={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")
        print(classification_report(y_test, y_pred,
                                    target_names=["Healthy","Moderate","Poor"],
                                    zero_division=0))

        if f1 > best_f1:
            best_f1, best_model_obj, best_name = f1, model, name

    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv(os.path.join(OUTPUT_DIR, "model_comparison_results.csv"), index=False)
    print(f"\n[ml_pipeline] Best model: {best_name} (weighted F1={best_f1:.4f})")

    # Save artefacts
    joblib.dump(best_model_obj, os.path.join(OUTPUT_DIR, "best_model.pkl"))
    joblib.dump(scaler,         os.path.join(OUTPUT_DIR, "scaler.pkl"))
    joblib.dump(X.columns.tolist(), os.path.join(OUTPUT_DIR, "feature_cols.pkl"))

    # Feature importance (only tree-based models have it)
    if hasattr(best_model_obj, "feature_importances_"):
        imp = pd.DataFrame({
            "feature":    X.columns.tolist(),
            "importance": best_model_obj.feature_importances_,
        }).sort_values("importance", ascending=False)
        imp.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
        print("\n[ml_pipeline] Top 8 features:")
        print(imp.head(8).to_string(index=False))

    return best_name, metrics_df


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    csv_path = os.path.join(DATA_DIR, CSV_FILES["sentinel"])
    try:
        X, y = prepare_ml_dataset(csv_path)
        best_name, metrics = train_and_serialize_models(X, y)
        print("\n=== Final Results ===")
        print(metrics.to_string(index=False))
    except Exception as e:
        print(f"[ERROR] {e}")
        raise
