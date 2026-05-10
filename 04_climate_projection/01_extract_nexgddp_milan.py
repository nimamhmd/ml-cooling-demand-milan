"""
============================================================================
EXTRACTION: NEX-GDDP-CMIP6 Daily Temperature for Milan

Extracts daily mean 2-metre air temperature at the Milan grid cell from
NASA NEX-GDDP-CMIP6 (Thrasher et al. 2022) for:
  - 12 GCMs (MPI-ESM1-2-HR, CMCC-ESM2, EC-Earth3, CNRM-ESM2-1, IPSL-CM6A-LR,
    CESM2, HadGEM3-GC31-LL, CanESM5, NorESM2-MM, GFDL-ESM4, MIROC6, ACCESS-ESM1-5)
  - 5 slices per model: historical 1990-2014, ssp245/ssp585 x 2030-2050/2080-2100
  - Total: 60 daily-temperature CSV files

Run this once to populate the input folder for downstream scripts.
Runtime: approximately 27 minutes (12 models x 5 slices).

Earth Engine collection: NASA/GDDP-CMIP6
Output folder: <project root>\Data Collections\Future (CMIP6) Climate Datasets\DATA
============================================================================
"""

import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime

try:
    import ee
except ImportError:
    print("Installing earthengine-api...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "earthengine-api"])
    import ee
try:
    import pandas as pd
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pandas"])
    import pandas as pd

CLOUD_PROJECT = "nima-21-11-2025"
ee.Initialize(project=CLOUD_PROJECT)

# Output folder (where the user keeps CMIP6 data collections)
CANDIDATE_OUTPUT_ROOTS = [
    Path(r"C:\Users\n.mohammadi\Desktop\NimaMohammadi\03. Nima Mohammadi - Thesis\Data Collections\Future (CMIP6) Climate Datasets\DATA"),
    Path(r"C:\Users\n.mohammadi\Desktop\NimaMohammadi\02.Nima Mohammadi - Thesis\Data Collections\Future (CMIP6) Climate Datasets\DATA"),
]
OUTPUT_DIR = next((p for p in CANDIDATE_OUTPUT_ROOTS if p.exists()), CANDIDATE_OUTPUT_ROOTS[0])
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Milan point (lat, lon) - centroid of urban area
MILAN_POINT = ee.Geometry.Point([9.19, 45.46])

MODELS = [
    "MPI-ESM1-2-HR", "CMCC-ESM2", "EC-Earth3", "CNRM-ESM2-1",
    "IPSL-CM6A-LR", "CESM2", "HadGEM3-GC31-LL", "CanESM5",
    "NorESM2-MM", "GFDL-ESM4", "MIROC6", "ACCESS-ESM1-5",
]

SLICES = [
    ("historical", 1990, 2014),
    ("ssp245", 2030, 2050),
    ("ssp245", 2080, 2100),
    ("ssp585", 2030, 2050),
    ("ssp585", 2080, 2100),
]


def extract_one(model, scenario, year_start, year_end):
    """Extract daily temperature for one (model, scenario, period) at Milan."""
    coll = (ee.ImageCollection("NASA/GDDP-CMIP6")
              .filter(ee.Filter.eq("model", model))
              .filter(ee.Filter.eq("scenario", scenario))
              .filterDate(f"{year_start}-01-01", f"{year_end+1}-01-01")
              .select("tas"))
    n = coll.size().getInfo()
    if n == 0:
        return None

    def to_feature(img):
        date = img.date().format("YYYY-MM-dd")
        val = img.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=MILAN_POINT,
            scale=27830,
            crs="EPSG:4326"
        ).get("tas")
        return ee.Feature(None, {"time": date, "tas": val})

    fc = coll.map(to_feature)
    rows = fc.getInfo()["features"]
    df = pd.DataFrame([f["properties"] for f in rows])
    df = df.dropna(subset=["tas"]).sort_values("time").reset_index(drop=True)
    return df


def main():
    print(f"Output: {OUTPUT_DIR}")
    print(f"Started: {datetime.now()}")
    t0 = time.time()
    total = len(MODELS) * len(SLICES)
    done = 0
    for model in MODELS:
        for scenario, y0, y1 in SLICES:
            done += 1
            fname = f"CMIP6_Milan_{model}_{scenario}_{y0}_{y1}.csv"
            out = OUTPUT_DIR / fname
            if out.exists():
                print(f"  [{done}/{total}] SKIP (exists): {fname}")
                continue
            print(f"  [{done}/{total}] Extracting: {fname}", end=" ... ")
            try:
                df = extract_one(model, scenario, y0, y1)
                if df is None or len(df) == 0:
                    print("EMPTY")
                    continue
                df.to_csv(out, index=False)
                print(f"OK ({len(df)} days)")
            except Exception as e:
                print(f"FAILED: {e}")
    print(f"\nDone in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
