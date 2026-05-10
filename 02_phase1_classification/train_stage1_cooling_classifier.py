import json
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

# =========================
# PATHS
# =========================
DATA_STAGE1 = Path(r"P:\Nima\23-11-2025\Data Collections\CENED\ml_dataset\prepared\stage1_classification_data.csv")

MODEL_DIR = Path(r"P:\Nima\23-11-2025\ML Models\20260217-Stage 1")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

OUT_MODEL = MODEL_DIR / "stage1_best_model.joblib"
OUT_METRICS = MODEL_DIR / "stage1_metrics.json"
OUT_FEATURES = MODEL_DIR / "stage1_feature_columns.json"

# =========================
# LOAD
# =========================
print("[1] Loading Stage 1 dataset...")
df = pd.read_csv(DATA_STAGE1, low_memory=False)
print("   Rows:", len(df), "Cols:", df.shape[1])

target = "y_cooling_present"
if target not in df.columns:
    raise ValueError(f"Missing target column: {target}")

# =========================
# FEATURE SELECTION (SAFE DEFAULT)
# =========================
# Remove obvious leakage/non-features:
DROP_COLS = [
    target,
    "CLIMATIZZAZIONE_ESTIVA",  # original boolean version of target
    "geometry",                # if present from GIS exports
]

# Also remove any columns that are basically IDs / text addresses (optional, but safer):
# We'll drop columns that often cause leakage or high-cardinality noise:
LEAKY_OR_ID_LIKE = [
    "cod_ape", "COD_APE", "indirizzo", "INDIRIZZO", "location", "LOCATION",
    "location_address", "location_city", "location_zip", "location_state",
]

drop_set = set([c for c in DROP_COLS + LEAKY_OR_ID_LIKE if c in df.columns])

X = df.drop(columns=list(drop_set), errors="ignore").copy()
y = df[target].astype(int).copy()

print("\n[2] X shape after drops:", X.shape)
print("   y distribution:\n", y.value_counts())

# Detect column types
num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

print("\n[3] Column types:")
print("   Numeric:", len(num_cols))
print("   Categorical:", len(cat_cols))

# =========================
# PREPROCESSING
# =========================
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ],
    remainder="drop"
)

# =========================
# TRAIN/TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n[4] Split:")
print("   Train:", X_train.shape, "Test:", X_test.shape)

# =========================
# MODELS
# =========================
models = {
    "logreg": LogisticRegression(max_iter=2000, class_weight="balanced"),
    "rf": RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    ),
}

results = {}

best_name = None
best_auc = -1
best_pipe = None

print("\n[5] Training candidates...")
for name, clf in models.items():
    pipe = Pipeline(steps=[
        ("preprocess", preprocess),
        ("model", clf)
    ])
    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    auc = roc_auc_score(y_test, proba)
    acc = accuracy_score(y_test, pred)
    prec = precision_score(y_test, pred, zero_division=0)
    rec = recall_score(y_test, pred, zero_division=0)
    f1 = f1_score(y_test, pred, zero_division=0)
    cm = confusion_matrix(y_test, pred).tolist()

    results[name] = {
        "AUC": float(auc),
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1": float(f1),
        "ConfusionMatrix": cm
    }

    print(f"   {name}: AUC={auc:.3f} Acc={acc:.3f} F1={f1:.3f}")

    if auc > best_auc:
        best_auc = auc
        best_name = name
        best_pipe = pipe

# =========================
# SAVE BEST
# =========================
print("\n[6] Best model:", best_name, "AUC:", round(best_auc, 3))

joblib.dump(best_pipe, OUT_MODEL)

# Save metrics + config
payload = {
    "best_model": best_name,
    "results": results,
    "data_path": str(DATA_STAGE1),
    "drop_columns": sorted(list(drop_set)),
    "n_rows": int(len(df)),
    "n_features_before_encoding": int(X.shape[1]),
    "n_numeric_cols": int(len(num_cols)),
    "n_categorical_cols": int(len(cat_cols)),
    "random_state": 42,
    "test_size": 0.2
}
OUT_METRICS.write_text(json.dumps(payload, indent=2), encoding="utf-8")

# Save feature column list (before encoding, for reproducibility)
OUT_FEATURES.write_text(json.dumps({"numeric": num_cols, "categorical": cat_cols}, indent=2), encoding="utf-8")

print("\n✅ DONE. Saved:")
print("   Model  :", OUT_MODEL)
print("   Metrics:", OUT_METRICS)
print("   Features:", OUT_FEATURES)
