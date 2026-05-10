from pathlib import Path
import pandas as pd
import os

# ---- PATHS (match your project folders) ----
BUILDINGS_WITH_EPC = Path(r"P:\Nima\23-11-2025\Data Collections\CENED\building_level_outputs\buildings_with_epc.csv")

# This file was created earlier in your ClimateProcessingProject
CLIMATE_CLIM = Path(r"P:\Nima\23-11-2025\PythonProjects\ClimateProcessingProject\data\ERA5_clean\climate_features_climatology_2005_2020.csv")

OUT_DIR = Path(r"P:\Nima\23-11-2025\Data Collections\CENED\ml_dataset")
OUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_PATH = OUT_DIR / "ml_dataset_buildings_epc_climate.csv"

print("[1] Loading buildings_with_epc...")
df = pd.read_csv(BUILDINGS_WITH_EPC, low_memory=False)
print("   Rows:", len(df), "Cols:", df.shape[1])

print("[2] Loading climate climatology (single row)...")
clim = pd.read_csv(CLIMATE_CLIM)

# drop non-feature columns if present
for c in ["period", "start_year", "end_year"]:
    if c in clim.columns:
        clim = clim.drop(columns=[c])

if len(clim) != 1:
    raise ValueError("Climate climatology file should contain exactly 1 row (a single baseline).")

print("   Climate columns:", list(clim.columns))

print("[3] Attaching climate columns to all buildings...")
for col in clim.columns:
    df[col] = clim.iloc[0][col]

print("[4] Saving final ML dataset...")
df.to_csv(OUT_PATH, index=False)

print("✅ DONE:", OUT_PATH)
