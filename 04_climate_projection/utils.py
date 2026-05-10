# utils.py
"""
Utility functions and shared configuration for the climate processing project.
You do NOT run this file directly. Other scripts import from it.
"""

from pathlib import Path
import glob

import numpy as np
import pandas as pd


# === PATHS FOR HISTORICAL ERA5 ===========================================

# Folder on F: where the raw ERA5 hourly CSVs are stored
RAW_ERA5_DIR = Path(
    r"F:\Building Engineering - Polytechnic University of Turin\Masters Thesis\23-11-2025\Data Collections\Climate Data Store\DATA\Google Earth Engine"
)

# Project root = folder where this file (utils.py) lives
PROJECT_ROOT = Path(__file__).resolve().parent

# Folders INSIDE your Python project where processed ERA5 data will be saved
ERA5_DAILY_DIR = PROJECT_ROOT / "data" / "ERA5_daily"
ERA5_CLEAN_DIR = PROJECT_ROOT / "data" / "ERA5_clean"

ERA5_DAILY_DIR.mkdir(parents=True, exist_ok=True)
ERA5_CLEAN_DIR.mkdir(parents=True, exist_ok=True)


# === FILE DISCOVERY (ERA5) ===============================================

def list_era5_hourly_files():
    """
    Return a sorted list of all ERA5 hourly CSV files for Milan.
    Expects filenames like: ERA5Land_Milan_1990_hourly.csv
    """
    pattern = RAW_ERA5_DIR / "ERA5Land_Milan_*_hourly.csv"
    paths = sorted(glob.glob(str(pattern)))
    return [Path(p) for p in paths]


# === UNIT CONVERSIONS & HELPERS ==========================================

def kelvin_to_celsius(series: pd.Series) -> pd.Series:
    """Convert temperature from Kelvin to Celsius."""
    return series - 273.15


def meters_to_mm(series: pd.Series) -> pd.Series:
    """Convert depth from metres to millimetres."""
    return series * 1000.0


def joule_to_watt(series: pd.Series, seconds: int = 3600) -> pd.Series:
    """
    Convert energy per square metre (J/m² over a period)
    to mean power (W/m²) over that period.
    For hourly ERA5 data: divide by 3600 seconds.
    """
    return series / float(seconds)


def wind_speed_from_components(u: pd.Series, v: pd.Series) -> pd.Series:
    """Compute wind speed magnitude from u and v components."""
    return np.sqrt(u**2 + v**2)


def parse_year_from_filename(path: Path) -> int:
    """
    Extract the year from a filename of the form:
    ERA5Land_Milan_1990_hourly.csv
    """
    name = path.name  # e.g. 'ERA5Land_Milan_1990_hourly.csv'
    parts = name.split("_")
    # ['ERA5Land', 'Milan', '1990', 'hourly.csv']
    year_str = parts[2]
    return int(year_str)


# === PATHS FOR FUTURE CLIMATE (CMIP6) ====================================

# Folder on F: where the raw CMIP6 CSVs are stored
RAW_CMIP6_DIR = Path(
    r"F:\Building Engineering - Polytechnic University of Turin\Masters Thesis\23-11-2025\Data Collections\Future (CMIP6) Climate Datasets\DATA"
)

# Folder INSIDE the Python project where cleaned CMIP6 data will be saved
CMIP6_CLEAN_DIR = PROJECT_ROOT / "data" / "CMIP6_clean"
CMIP6_CLEAN_DIR.mkdir(parents=True, exist_ok=True)
