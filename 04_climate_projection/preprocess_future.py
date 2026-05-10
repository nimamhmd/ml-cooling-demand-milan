# preprocess_future.py
"""
Preprocess CMIP6 future climate datasets for Milan (SSP2-4.5 and SSP5-8.5).

Input:
    F:\Building Engineering - Polytechnic University of Turin\Masters Thesis\23-11-2025\
      Data Collections\Future (CMIP6) Climate Datasets\DATA\
        CMIP6_Milan_ssp245_2030_2050.csv
        CMIP6_Milan_ssp245_2080_2100.csv
        CMIP6_Milan_ssp585_2030_2050.csv
        CMIP6_Milan_ssp585_2080_2100.csv

Output (cleaned + ML-friendly):
    <project folder>\data\CMIP6_clean\
        CMIP6_Milan_ssp245_2030_2050_clean.csv
        CMIP6_Milan_ssp245_2080_2100_clean.csv
        CMIP6_Milan_ssp585_2030_2050_clean.csv
        CMIP6_Milan_ssp585_2080_2100_clean.csv

Each output file:
    - has a clean datetime column "time"
    - removes 'system:index' and '.geo'
    - keeps original CMIP6 units (tas, pr, etc.)
    - adds derived variables:
        tas_C, tasmax_C, tasmin_C  (°C)
        pr_mm_day                  (mm/day)
"""

import pandas as pd
from pathlib import Path

from utils import RAW_CMIP6_DIR, CMIP6_CLEAN_DIR


def clean_cmip6_file(path: Path) -> Path:
    """Load one CMIP6 CSV, standardize, add derived vars, and save a clean version."""
    print(f"Processing: {path.name}")

    df = pd.read_csv(path)

    # --- basic cleaning ----------------------------------------------------
    # drop junk columns if present
    for col in ["system:index", ".geo"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # ensure time column is datetime and sorted
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time")

    # --- metadata from filename -------------------------------------------
    # example stem: CMIP6_Milan_ssp245_2030_2050
    stem_parts = path.stem.split("_")
    # ['CMIP6', 'Milan', 'ssp245', '2030', '2050']
    scenario = stem_parts[2]       # ssp245 / ssp585
    period_start = int(stem_parts[3])
    period_end = int(stem_parts[4])

    df["scenario"] = scenario
    df["period_start"] = period_start
    df["period_end"] = period_end

    # --- derived variables (keep originals + add new) ---------------------
    # temperature: Kelvin -> Celsius
    if "tas" in df.columns:
        df["tas_C"] = df["tas"] - 273.15
    if "tasmax" in df.columns:
        df["tasmax_C"] = df["tasmax"] - 273.15
    if "tasmin" in df.columns:
        df["tasmin_C"] = df["tasmin"] - 273.15

    # precipitation: kg/m2/s -> mm/day  (1 kg/m2 = 1 mm)
    if "pr" in df.columns:
        df["pr_mm_day"] = df["pr"] * 86400.0

    # --- reorder columns: time first, then climate vars, then meta --------
    meta_cols = ["scenario", "period_start", "period_end"]
    cols = list(df.columns)

    # force 'time' to be first if present
    if "time" in cols:
        cols.remove("time")
        cols.insert(0, "time")

    # ensure meta columns are at the end
    for c in meta_cols:
        if c in cols:
            cols.remove(c)
            cols.append(c)

    df = df[cols]

    # --- save output -------------------------------------------------------
    out_name = f"CMIP6_Milan_{scenario}_{period_start}_{period_end}_clean.csv"
    out_path = CMIP6_CLEAN_DIR / out_name
    df.to_csv(out_path, index=False)

    print(f"Saved → {out_path}\n")
    return out_path


def main():
    print("RAW_CMIP6_DIR:", RAW_CMIP6_DIR)

    files = sorted(RAW_CMIP6_DIR.glob("CMIP6_Milan_*.csv"))

    if not files:
        print("❌ No CMIP6 files found in the folder!")
        return

    print(f"Found {len(files)} CMIP6 files.\n")

    out_paths = []
    for f in files:
        out_paths.append(clean_cmip6_file(f))

    print("Clean CMIP6 files generated:")
    for p in out_paths:
        print("  -", p)

    print("\nCMIP6 preprocessing completed successfully!")


if __name__ == "__main__":
    main()
