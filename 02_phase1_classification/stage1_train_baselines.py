# ============================================================
# STAGE 1 – Cooling Adoption Classification (Baselines)
# Project: Milan Cooling Demand Thesis
#
# What this script does:
# 1) Auto-detects the latest spatial_join_v2 output folder and loads:
#       buildings_with_epc_v2.csv
# 2) Prints cooling label distribution (critical sanity check)
# 3) Builds a leakage-safe feature set for Stage 1
# 4) Trains baseline models:
#       - Logistic Regression
#       - Random Forest
#       - HistGradientBoostingClassifier (fast strong baseline)
# 5) Saves metrics + artifacts into:
#       ML_Models/Stage1/run_<timestamp>/
#
# If auto-detection fails, you only need to paste ONE file path.
# ============================================================

import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
import joblib


# ---------------------------
# CONFIG
# ---------------------------
LABEL_COL = "CLIMATIZZAZIONE_ESTIVA"     # target source column from EPC
TARGET_COL = "y_cooling_present"         # internal target name
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Keep nearest join cutoff conservative already done (<=50m) in your spatial join v2.
# Here we focus on preventing leakage and robust validation.

# Columns to drop for leakage / invalid predictors (rule-based)
DROP_IF_CONTAINS = [
    "CE_",                       # cooling system / cooling energy fields
    "RAFFRESC",                  # Italian stem for cooling (raffrescamento)
    "CLIMATIZZAZIONE_ESTIVA",    # target itself
]

# Some known leakage columns (drop if present)
DROP_EXACT = {
    "SUPERF_UTILE_RAFFRESCATA",
    "VOLUME_LORDO_RAFFRESCATO",
    "EP_RAFFRESCAMENTO",         # just in case
    "EER", "SEER",               # possible cooling performance indicators
}

# Climate columns you likely want (if present)
PREFERRED_CLIMATE = [
    "CDD26", "mean_temp_summer", "max_temp", "RH_mean", "HI_mean",
    "ts_anom", "ts_anom_summer", "mean_temp"
]

# A few safe building geometry/type columns often present (use if present)
PREFERRED_BUILDING = [
    "area_m2_x", "building_use_x", "building_type_x", "building_status_x",
    "construction_year", "anno_costruzione", "year_built",
    "VOLUME", "volume", "volume_m3"
]


def stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def find_latest_buildings_with_epc_v2(project_root: Path) -> Path | None:
    """
    Auto-detect the latest:
      data/01_processed/spatial_join_v2/run_*/buildings_with_epc_v2.csv
    """
    base = project_root / "data" / "01_processed" / "spatial_join_v2"
    if not base.exists():
        return None

    candidates = list(base.glob("run_*/buildings_with_epc_v2.csv"))
    if not candidates:
        return None

    # Choose newest by folder modified time
    candidates_sorted = sorted(
        candidates,
        key=lambda p: p.parent.stat().st_mtime,
        reverse=True
    )
    return candidates_sorted[0]


def choose_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """
    Decide numeric + categorical features safely.
    - Drop leakage columns.
    - Prefer climate + building cols if present.
    - If not, use all non-target columns after leakage filter.
    Returns: (numeric_cols, categorical_cols, dropped_cols)
    """
    cols = list(df.columns)

    dropped = set()

    # Drop exact leakage
    for c in cols:
        if c in DROP_EXACT:
            dropped.add(c)

    # Drop columns containing leakage stems
    for c in cols:
        for token in DROP_IF_CONTAINS:
            if token.lower() in str(c).lower():
                dropped.add(c)

    # Drop geometry columns (not ML-friendly unless engineered)
    for gcol in ["geometry", "GEOMETRY", "wkt", "WKT"]:
        if gcol in cols:
            dropped.add(gcol)

    # Determine candidate feature columns
    feature_cols = [c for c in cols if c not in dropped]

    # Remove obvious identifiers that may cause leakage/memorization if present
    for idc in ["FEATURE_ID", "building_id", "id", "ID", "fid", "FID"]:
        if idc in feature_cols:
            feature_cols.remove(idc)
            dropped.add(idc)

    # Must remove target column if still present
    if TARGET_COL in feature_cols:
        feature_cols.remove(TARGET_COL)

    # If label col still present, remove (should be in dropped already)
    if LABEL_COL in feature_cols:
        feature_cols.remove(LABEL_COL)

    # Prefer a meaningful subset if available (safer + more stable)
    preferred = []
    for c in PREFERRED_BUILDING + PREFERRED_CLIMATE:
        if c in feature_cols:
            preferred.append(c)

    # If preferred list is too small, fall back to all remaining feature_cols
    if len(preferred) >= 8:
        use_cols = preferred
    else:
        use_cols = feature_cols

    # Split numeric vs categorical
    numeric_cols = []
    categorical_cols = []
    for c in use_cols:
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    return numeric_cols, categorical_cols, sorted(list(dropped))


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_test)[:, 1]
    pred = model.predict(X_test)

    out = {
        "model": name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }
    if proba is not None:
        try:
            out["roc_auc"] = float(roc_auc_score(y_test, proba))
        except Exception:
            out["roc_auc"] = None
    else:
        out["roc_auc"] = None

    out["classification_report"] = classification_report(y_test, pred, zero_division=0)
    return out


def main():
    project_root = Path(__file__).resolve().parents[1]

    # Output folder for this run
    out_dir = project_root / "ML_Models" / "Stage1" / f"run_{stamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Auto-detect input
    data_path = find_latest_buildings_with_epc_v2(project_root)

    if data_path is None:
        print("❌ Could not auto-find buildings_with_epc_v2.csv.")
        print("Please set DATA_PATH manually inside this script.")
        return

    print("✅ Using dataset:", data_path)

    # 2) Load
    df = pd.read_csv(data_path, low_memory=False)
    print("Loaded shape:", df.shape)

    if LABEL_COL not in df.columns:
        print(f"❌ Label column '{LABEL_COL}' not found in dataset.")
        print("Columns:", list(df.columns)[:50], "...")
        return

    # 3) Build target
    labeled = df[df[LABEL_COL].notna()].copy()
    print("\n--- Label coverage ---")
    print("Total buildings:", len(df))
    print("Labeled buildings:", len(labeled))
    print("Labeled share:", round(len(labeled) / max(len(df), 1), 4))

    # Normalize label to 0/1
    # In your data it might already be 0/1 or True/False or 'SI'/'NO'
    def norm_label(x):
        if pd.isna(x):
            return np.nan
        s = str(x).strip().lower()
        if s in ["1", "true", "t", "si", "sì", "yes", "y"]:
            return 1
        if s in ["0", "false", "f", "no", "n"]:
            return 0
        # if numeric-like
        try:
            v = float(s.replace(",", "."))
            return 1 if v >= 1 else 0
        except Exception:
            return np.nan

    labeled[TARGET_COL] = labeled[LABEL_COL].apply(norm_label)

    # Drop rows where label couldn't be normalized
    labeled = labeled[labeled[TARGET_COL].isin([0, 1])].copy()

    print("\n--- Cooling distribution (Stage 1 target) ---")
    vc = labeled[TARGET_COL].value_counts(dropna=False)
    print(vc)
    print("Shares:\n", labeled[TARGET_COL].value_counts(normalize=True))

    # 4) Feature selection (leakage-safe)
    numeric_cols, categorical_cols, dropped_cols = choose_feature_columns(labeled)

    # Save feature selection log
    (out_dir / "dropped_columns.txt").write_text("\n".join(dropped_cols), encoding="utf-8")
    (out_dir / "numeric_features.txt").write_text("\n".join(numeric_cols), encoding="utf-8")
    (out_dir / "categorical_features.txt").write_text("\n".join(categorical_cols), encoding="utf-8")

    print("\n--- Feature set ---")
    print("Numeric features:", len(numeric_cols))
    print("Categorical features:", len(categorical_cols))
    print("Total used features:", len(numeric_cols) + len(categorical_cols))
    print("Feature lists saved in:", out_dir)

    # Prepare X/y
    used_cols = numeric_cols + categorical_cols
    X = labeled[used_cols].copy()
    y = labeled[TARGET_COL].astype(int).copy()

    # 5) Train/test split (stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
    )

    # 6) Preprocessing
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop"
    )

    # 7) Models
    models = {
        "logreg": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",   # helps if imbalanced
            n_jobs=None
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            random_state=RANDOM_STATE,
            class_weight="balanced_subsample",
            n_jobs=-1
        ),
        "hist_gb": HistGradientBoostingClassifier(
            random_state=RANDOM_STATE
        )
    }

    results = []
    best_model_name = None
    best_auc = -1

    for name, clf in models.items():
        pipe = Pipeline(steps=[
            ("preprocess", preprocessor),
            ("model", clf)
        ])

        print(f"\n=== Training: {name} ===")
        pipe.fit(X_train, y_train)

        metrics = evaluate_model(name, pipe, X_test, y_test)
        results.append(metrics)

        # Save model
        joblib.dump(pipe, out_dir / f"model_{name}.joblib")

        # Track best by ROC-AUC if available, else F1
        auc = metrics.get("roc_auc")
        if auc is not None and auc > best_auc:
            best_auc = auc
            best_model_name = name

    # 8) Save results
    (out_dir / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Also write a human-readable summary
    lines = []
    for r in results:
        lines.append(f"Model: {r['model']}")
        lines.append(f"  Accuracy : {r['accuracy']:.4f}")
        lines.append(f"  Precision: {r['precision']:.4f}")
        lines.append(f"  Recall   : {r['recall']:.4f}")
        lines.append(f"  F1       : {r['f1']:.4f}")
        lines.append(f"  ROC-AUC  : {r['roc_auc'] if r['roc_auc'] is not None else 'N/A'}")
        lines.append(f"  Confusion: {r['confusion_matrix']}")
        lines.append("")
    if best_model_name:
        lines.append(f"Best model by ROC-AUC: {best_model_name} (AUC={best_auc:.4f})")
    (out_dir / "metrics_summary.txt").write_text("\n".join(lines), encoding="utf-8")

    print("\n✅ Stage 1 baseline training complete.")
    print("Outputs saved to:", out_dir)
    if best_model_name:
        print(f"Best model so far: {best_model_name} (ROC-AUC={best_auc:.4f})")
    print("\nNext: we will (1) tune the best model, (2) calibrate probabilities, (3) generate citywide cooling probability for all 53,041 buildings.")


if __name__ == "__main__":
    main()