# preprocess_historical.py
"""
Preprocess ERA5-Land hourly climate data (1990–2024) for Milan.

This script:
- loads each ERA5 hourly CSV from the raw data folder
- converts units (Kelvin → Celsius, J/m² → W/m², m → mm)
- computes daily averages
- saves one daily CSV per year into data/ERA5_daily/
- merges all years into a single dataset in data/ERA5_clean/
"""

import pandas as pd
from pathlib import Path

from utils import (
    list_era5_hourly_files,
    kelvin_to_celsius,
    joule_to_watt,
    meters_to_mm,
    wind_speed_from_components,
    parse_year_from_filename,
    ERA5_DAILY_DIR,
    ERA5_CLEAN_DIR,
)


def process_single_year(path: Path) -> pd.DataFrame:
    """Load one ERA5 hourly CSV and return a cleaned DAILY dataset."""
    print(f"Processing: {path.name}")

    df = pd.read_csv(path)

    # Convert time to datetime
    df["time"] = pd.to_datetime(df["time"])

    # ---- Temperature conversions (K → °C) ----
    if "temperature_2m" in df:
        df["temperature_2m"] = kelvin_to_celsius(df["temperature_2m"])
    if "dewpoint_temperature_2m" in df:
        df["dewpoint_temperature_2m"] = kelvin_to_celsius(df["dewpoint_temperature_2m"])
    if "skin_temperature" in df:
        df["skin_temperature"] = kelvin_to_celsius(df["skin_temperature"])

    # ---- Radiation conversions (J/m² → W/m²) ----
    radiation_cols = [
        "surface_solar_radiation_downwards",
        "surface_net_solar_radiation",
        "surface_thermal_radiation_downwards",
        "surface_net_thermal_radiation",
    ]
    for col in radiation_cols:
        if col in df:
            df[col] = joule_to_watt(df[col])

    # ---- Precipitation (m → mm) ----
    if "total_precipitation" in df:
        df["total_precipitation"] = meters_to_mm(df["total_precipitation"])

    # ---- Wind speed (from u and v components) ----
    if "u_component_of_wind_10m" in df and "v_component_of_wind_10m" in df:
        df["wind_speed_10m"] = wind_speed_from_components(
            df["u_component_of_wind_10m"], df["v_component_of_wind_10m"]
        )

    # ---- Daily aggregation ----
    df_daily = df.resample("D", on="time").mean(numeric_only=True)

    # Add year column for convenience
    df_daily["year"] = df_daily.index.year

    return df_daily


def save_daily_files():
    """Process all ERA5 files and save daily CSVs for each year."""
    ERA5_DAILY_DIR.mkdir(parents=True, exist_ok=True)

    files = list_era5_hourly_files()
    print("Found", len(files), "ERA5 files.")

    for f in files:
        year = parse_year_from_filename(f)
        df_daily = process_single_year(f)

        out_path = ERA5_DAILY_DIR / f"ERA5_Milan_{year}_daily.csv"
        df_daily.to_csv(out_path, index=True)
        print("Saved:", out_path)


def merge_all_daily_files():
    """Merge all daily files into one master dataset."""
    ERA5_CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    daily_files = sorted(ERA5_DAILY_DIR.glob("ERA5_Milan_*_daily.csv"))
    dfs = [pd.read_csv(f, parse_dates=["time"]) for f in daily_files]

    df_all = pd.concat(dfs).sort_values("time")

    out_path = ERA5_CLEAN_DIR / "ERA5_Milan_1990_2024_daily.csv"
    df_all.to_csv(out_path, index=False)

    print("Merged dataset saved to:", out_path)
    print("Shape:", df_all.shape)


def main():
    save_daily_files()
    merge_all_daily_files()
    print("\nERA5 preprocessing completed successfully!")


if __name__ == "__main__":
    main()
