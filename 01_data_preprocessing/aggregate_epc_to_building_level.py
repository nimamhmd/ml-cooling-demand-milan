import os
import pandas as pd

# =========================
# PATHS
# =========================
JOINED_CSV = r"P:\Nima\23-11-2025\Data Collections\CENED\spatial_join_outputs\epc_joined_to_buildings.csv"
OUT_DIR = r"P:\Nima\23-11-2025\Data Collections\CENED\spatial_join_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_BEST_CSV = os.path.join(OUT_DIR, "epc_best_per_building.csv")
OUT_BEST_PARQUET = os.path.join(OUT_DIR, "epc_best_per_building.parquet")

# =========================
# LOAD
# =========================
df = pd.read_csv(JOINED_CSV, low_memory=False)
print("Loaded rows:", len(df))
print("Columns:", list(df.columns))

# =========================
# IDENTIFY BUILDING ID COLUMN
# =========================
# Common candidates based on your buildings pipeline
building_candidates = ["building_id", "EDIFC_ID", "FEATURE_ID", "ID_ZRIL"]

building_col = next((c for c in building_candidates if c in df.columns), None)
if building_col is None:
    raise ValueError("No building identifier column found. Check your joined CSV columns.")

print("Detected building id column:", building_col)

# Keep only matched rows (drop those still unmatched)
# Depending on your sjoin output, index_right may exist
if "index_right" in df.columns:
    before = len(df)
    df = df[df["index_right"].notna()].copy()
    print(f"Removed unmatched rows: {before} -> {len(df)}")

# =========================
# DETECT DATE COLUMN (IF EXISTS)
# =========================
date_candidates = [
    "DATA_EMISSIONE", "DATA_RILASCIO", "DATA_APE", "DATA_CERTIFICATO", "DATA",
    "data_emissione", "data_rilascio", "data_ape"
]
year_candidates = ["ANNO", "ANNO_APE", "ANNO_EMISSIONE", "anno", "year"]

date_col = next((c for c in date_candidates if c in df.columns), None)
year_col = next((c for c in year_candidates if c in df.columns), None)

print("Detected date column:", date_col)
print("Detected year column:", year_col)

# =========================
# RULE 1 (Preferred): LATEST CERTIFICATE DATE
# =========================
if date_col:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values([building_col, date_col], ascending=[True, False])
    best = df.drop_duplicates(subset=[building_col], keep="first").copy()
    rule_used = f"latest_by_{date_col}"

# =========================
# RULE 2 (Fallback): LATEST YEAR
# =========================
elif year_col:
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
    df = df.sort_values([building_col, year_col], ascending=[True, False])
    best = df.drop_duplicates(subset=[building_col], keep="first").copy()
    rule_used = f"latest_by_{year_col}"

# =========================
# RULE 3 (Fallback): MOST COMPLETE ROW
# =========================
else:
    df["non_null_count"] = df.notna().sum(axis=1)
    df = df.sort_values([building_col, "non_null_count"], ascending=[True, False])
    best = df.drop_duplicates(subset=[building_col], keep="first").copy()
    best = best.drop(columns=["non_null_count"])
    rule_used = "most_complete_row"

print("\nAggregation rule used:", rule_used)
print("Unique buildings with EPC:", best[building_col].nunique())
print("Best EPC rows saved:", len(best))

# =========================
# SAVE OUTPUTS
# =========================
best.to_csv(OUT_BEST_CSV, index=False)
best.to_parquet(OUT_BEST_PARQUET, index=False)

print("\nSaved:", OUT_BEST_CSV)
print("Saved:", OUT_BEST_PARQUET)
print("\n✅ DONE: One EPC row per building created.")

