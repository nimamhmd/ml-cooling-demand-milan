import pandas as pd
from pathlib import Path

# =========================
# PATHS
# =========================
ML_DATA = Path(r"P:\Nima\23-11-2025\Data Collections\CENED\ml_dataset\ml_dataset_buildings_epc_climate.csv")

# Data-prep outputs stay near the dataset (recommended)
OUT_DIR = Path(r"P:\Nima\23-11-2025\Data Collections\CENED\ml_dataset\prepared")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_STAGE1 = OUT_DIR / "stage1_classification_data.csv"
OUT_STAGE2 = OUT_DIR / "stage2_regression_data.csv"
OUT_INFER  = OUT_DIR / "inference_no_epc.csv"

print("[1] Loading ML dataset...")
df = pd.read_csv(ML_DATA, low_memory=False)
print("   Rows:", len(df), "Cols:", df.shape[1])

# =========================
# TARGET COLUMNS (as in your dataset)
# =========================
col_cool = "CLIMATIZZAZIONE_ESTIVA"
col_area = "SUPERF_UTILE_RAFFRESCATA"

if col_cool not in df.columns:
    raise ValueError(f"Missing column: {col_cool}")
if col_area not in df.columns:
    raise ValueError(f"Missing column: {col_area}")

print("\n[2] Target distribution (raw):")
print(df[col_cool].value_counts(dropna=False))

# =========================
# STAGE 1 DATASET
# =========================
print("\n[3] Preparing Stage 1 dataset (cooling True/False only)...")
stage1 = df[df[col_cool].notna()].copy()

# Convert boolean -> 0/1
stage1["y_cooling_present"] = stage1[col_cool].astype(bool).astype(int)

print("   Stage 1 rows:", len(stage1))
print("   y_cooling_present distribution:")
print(stage1["y_cooling_present"].value_counts())

stage1.to_csv(OUT_STAGE1, index=False)
print("   Saved:", OUT_STAGE1)

# =========================
# STAGE 2 DATASET (later)
# =========================
print("\n[4] Preparing Stage 2 dataset (cooling TRUE only)...")
stage2 = stage1[stage1["y_cooling_present"] == 1].copy()

# Keep only valid cooled-area values
stage2 = stage2[stage2[col_area].notna()].copy()
stage2 = stage2[stage2[col_area] > 0].copy()

stage2["y_cooled_area_m2"] = stage2[col_area]

print("   Stage 2 rows (cooling=True & area>0):", len(stage2))
print("   cooled area stats:")
print(stage2["y_cooled_area_m2"].describe())

stage2.to_csv(OUT_STAGE2, index=False)
print("   Saved:", OUT_STAGE2)

# =========================
# INFERENCE SET (NO EPC)
# =========================
print("\n[5] Preparing inference dataset (cooling unknown = NaN)...")
infer = df[df[col_cool].isna()].copy()
print("   Inference rows:", len(infer))

infer.to_csv(OUT_INFER, index=False)
print("   Saved:", OUT_INFER)

print("\n✅ DONE. Next: Stage 1 ML training on stage1_classification_data.csv")
