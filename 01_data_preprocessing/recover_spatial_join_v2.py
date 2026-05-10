# ============================================================
# MILAN COOLING THESIS – RECOVER MORE EPC↔DBT MATCHES (v2)
# Author: Nima Mohammadi
#
# Purpose:
#   Improve spatial join coverage between EPC points (CENED) and
#   DBT building polygons by:
#     1) within join
#     2) intersects join
#     3) nearest join with distance thresholds
#   Then select best EPC per building and output quality reports.
#
# Outputs (timestamped folder):
#   data/01_processed/spatial_join_v2/run_<timestamp>/
# ============================================================

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd

# Geo stack
import geopandas as gpd
from shapely.geometry import Point

# -----------------------------
# USER CONFIG (EDIT THESE)
# -----------------------------

# 1) EPC cleaned CSV (must include lat/lon columns)
EPC_CSV = r"P:\Nima\23-11-2025\Data Collections\CENED\ml_dataset\epc_clean_final.csv"

# 2) DBT buildings polygons (SHP or GPKG)
# If GPKG, you may need the layer name. Set DBT_LAYER to None for SHP.
DBT_FILE = r"P:\Nima\23-11-2025\Data Collections\DBT2012\buildings.gpkg"
DBT_LAYER = None  # e.g. "edifici" if needed, else None

# 3) Output root (recommended inside your project)
OUTPUT_ROOT = r"P:\Nima\23-11-2025\PythonProjects\CENED_Project\data\01_processed\spatial_join_v2"

# 4) Nearest join distance thresholds (meters) – conservative → wider
NEAREST_THRESHOLDS_M = [10, 25, 50, 100]

# 5) Optional: filter DBT to residential only (leave None to keep all)
# If you know residential codes, put them here (example codes are placeholders):
RESIDENTIAL_FILTER = None
# Example:
# RESIDENTIAL_FILTER = {"building_use_x": ["201", 201, "RES", "Residential"]}

# -----------------------------
# Column detection helpers
# -----------------------------
LAT_CANDIDATES = ["lat", "latitude", "y", "coord_y", "latitudine", "centroid_lat", "LAT", "Latitude"]
LON_CANDIDATES = ["lon", "lng", "longitude", "x", "coord_x", "longitudine", "centroid_lon", "LON", "Longitude"]

DATE_CANDIDATES = ["data_attestato", "data", "date", "DATA", "DATA_ATTESTATO", "DATA_EMISSIONE", "created_at"]

# Building ID candidates inside DBT polygons
BUILDING_ID_CANDIDATES = [
    "building_id", "BUILDING_ID", "id", "ID", "fid", "FID",
    "FEATURE_ID", "FEATURE_ID_x", "ID_ZRIL", "ID_ZRIL_x"
]


def stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def find_first_existing(col_list, df_cols):
    df_cols_lower = {c.lower(): c for c in df_cols}
    for c in col_list:
        if c.lower() in df_cols_lower:
            return df_cols_lower[c.lower()]
    return None


def pick_building_id_column(gdf: gpd.GeoDataFrame) -> str:
    cols = list(gdf.columns)
    for c in BUILDING_ID_CANDIDATES:
        if c in cols:
            return c
    # fallback: first non-geometry column
    for c in cols:
        if c != "geometry":
            return c
    raise ValueError("Could not detect a building ID column in DBT polygons.")


def parse_best_date(df: pd.DataFrame, date_col: str) -> pd.Series:
    # robust date parsing
    return pd.to_datetime(df[date_col], errors="coerce", dayfirst=True, infer_datetime_format=True)


def best_epc_per_building(matched: pd.DataFrame, building_id_col: str) -> pd.DataFrame:
    """
    Choose best EPC per building.
    Priority:
      1) Most recent date (if any detected)
      2) Most complete row (fewest NaNs)
    """
    # detect a date column
    date_col = None
    for c in matched.columns:
        if c in DATE_CANDIDATES:
            date_col = c
            break
    if date_col is None:
        # try loose match by lowercase contains
        for c in matched.columns:
            if any(dc.lower() in c.lower() for dc in DATE_CANDIDATES):
                date_col = c
                break

    m = matched.copy()
    m["_nan_count"] = m.isna().sum(axis=1)

    if date_col is not None:
        m["_date_parsed"] = parse_best_date(m, date_col)
    else:
        m["_date_parsed"] = pd.NaT

    # Sort:
    # - fewer NaNs is better
    # - more recent date is better
    m = m.sort_values(
        by=[building_id_col, "_nan_count", "_date_parsed"],
        ascending=[True, True, False],
        kind="mergesort"
    )

    best = m.groupby(building_id_col, as_index=False).head(1).drop(columns=["_nan_count", "_date_parsed"])
    return best


def main():
    run_id = f"run_{stamp()}"
    out_dir = os.path.join(OUTPUT_ROOT, run_id)
    safe_mkdir(out_dir)

    print("============================================================")
    print("RECOVER SPATIAL JOIN v2 – START")
    print("Run:", run_id)
    print("Output:", out_dir)
    print("============================================================\n")

    # -----------------------------
    # Load EPC CSV
    # -----------------------------
    if not os.path.exists(EPC_CSV):
        raise FileNotFoundError(f"EPC_CSV not found: {EPC_CSV}")

    epc = pd.read_csv(EPC_CSV, low_memory=False)
    print(f"✅ EPC loaded: {epc.shape}")

    lat_col = find_first_existing(LAT_CANDIDATES, epc.columns)
    lon_col = find_first_existing(LON_CANDIDATES, epc.columns)
    if lat_col is None or lon_col is None:
        raise ValueError(
            "Could not detect lat/lon columns in EPC CSV.\n"
            f"Detected lat={lat_col}, lon={lon_col}. "
            "Please rename or add your column names to LAT_CANDIDATES/LON_CANDIDATES."
        )

    # Build EPC GeoDataFrame in WGS84
    epc_geo = epc.copy()
    epc_geo = epc_geo[epc_geo[lat_col].notna() & epc_geo[lon_col].notna()].copy()
    epc_geo["geometry"] = [Point(xy) for xy in zip(epc_geo[lon_col], epc_geo[lat_col])]
    epc_gdf = gpd.GeoDataFrame(epc_geo, geometry="geometry", crs="EPSG:4326")
    print(f"✅ EPC with geometry: {epc_gdf.shape}")

    # -----------------------------
    # Load DBT polygons
    # -----------------------------
    if not os.path.exists(DBT_FILE):
        raise FileNotFoundError(f"DBT_FILE not found: {DBT_FILE}")

    if DBT_FILE.lower().endswith(".gpkg"):
        if DBT_LAYER is None:
            # try first layer
            layers = fiona_listlayers(DBT_FILE)
            if not layers:
                raise ValueError("No layers found in GPKG.")
            layer_to_use = layers[0]
            print(f"ℹ GPKG layers found: {layers}")
            print(f"ℹ Using first layer: {layer_to_use}")
        else:
            layer_to_use = DBT_LAYER
        dbt = gpd.read_file(DBT_FILE, layer=layer_to_use)
    else:
        dbt = gpd.read_file(DBT_FILE)

    print(f"✅ DBT polygons loaded: {dbt.shape}")

    if dbt.crs is None:
        raise ValueError("DBT polygons CRS is missing. Please set it in the GIS file (QGIS) and re-run.")

    # Optional filter to residential
    if RESIDENTIAL_FILTER:
        for k, allowed in RESIDENTIAL_FILTER.items():
            if k in dbt.columns:
                before = len(dbt)
                dbt = dbt[dbt[k].isin(allowed)].copy()
                print(f"ℹ Residential filter on {k}: {before} → {len(dbt)}")
            else:
                print(f"⚠ Residential filter column not found in DBT: {k}")

    building_id_col = pick_building_id_column(dbt)
    print(f"✅ Using building ID column: {building_id_col}")

    # Project EPC to DBT CRS for meter distances
    epc_proj = epc_gdf.to_crs(dbt.crs)

    # -----------------------------
    # Step 1: WITHIN join
    # -----------------------------
    print("\n--- Step 1: Spatial join (within) ---")
    within = gpd.sjoin(epc_proj, dbt[[building_id_col, "geometry"]], how="left", predicate="within")
    within["match_method"] = np.where(within[building_id_col].notna(), "within", None)

    matched_within = within[within[building_id_col].notna()].copy()
    unmatched = within[within[building_id_col].isna()].drop(columns=[building_id_col, "index_right"], errors="ignore").copy()

    print(f"Matched (within): {len(matched_within)}")
    print(f"Unmatched after within: {len(unmatched)}")

    # -----------------------------
    # Step 2: INTERSECTS join for boundary cases
    # -----------------------------
    print("\n--- Step 2: Spatial join (intersects) on unmatched ---")
    if len(unmatched) > 0:
        unmatched_gdf = gpd.GeoDataFrame(unmatched, geometry="geometry", crs=dbt.crs)
        inter = gpd.sjoin(unmatched_gdf, dbt[[building_id_col, "geometry"]], how="left", predicate="intersects")
        inter["match_method"] = np.where(inter[building_id_col].notna(), "intersects", None)

        matched_inter = inter[inter[building_id_col].notna()].copy()
        unmatched2 = inter[inter[building_id_col].isna()].drop(columns=[building_id_col, "index_right"], errors="ignore").copy()

        print(f"Matched (intersects): {len(matched_inter)}")
        print(f"Unmatched after intersects: {len(unmatched2)}")
    else:
        matched_inter = matched_within.iloc[0:0].copy()
        unmatched2 = unmatched.copy()

    # -----------------------------
    # Step 3: NEAREST join with thresholds
    # -----------------------------
    print("\n--- Step 3: Nearest join (thresholded) ---")
    matched_nearest_parts = []
    current_unmatched = unmatched2

    for thr in NEAREST_THRESHOLDS_M:
        if len(current_unmatched) == 0:
            break

        cur_gdf = gpd.GeoDataFrame(current_unmatched, geometry="geometry", crs=dbt.crs)

        nearest = gpd.sjoin_nearest(
            cur_gdf,
            dbt[[building_id_col, "geometry"]],
            how="left",
            distance_col="nearest_dist_m",
            max_distance=thr
        )
        got = nearest[nearest[building_id_col].notna()].copy()
        got["match_method"] = f"nearest_{thr}m"

        # keep only those matched at this threshold
        matched_nearest_parts.append(got)

        # remove matched from unmatched
        still_unmatched = nearest[nearest[building_id_col].isna()].drop(columns=[building_id_col, "index_right"], errors="ignore").copy()
        current_unmatched = still_unmatched

        print(f"Threshold {thr}m: matched {len(got)}, remaining unmatched {len(current_unmatched)}")

    matched_nearest = pd.concat(matched_nearest_parts, ignore_index=True) if matched_nearest_parts else matched_within.iloc[0:0].copy()
    final_unmatched = current_unmatched

    # -----------------------------
    # Combine matches and output join table
    # -----------------------------
    matched_all = pd.concat([matched_within, matched_inter, matched_nearest], ignore_index=True)

    # Match summary
    summary = {
        "epc_total_rows": int(len(epc)),
        "epc_with_coords": int(len(epc_gdf)),
        "dbt_buildings": int(len(dbt)),
        "matches_within": int(len(matched_within)),
        "matches_intersects": int(len(matched_inter)),
        "matches_nearest": int(len(matched_nearest)),
        "total_matched_epc_rows": int(len(matched_all)),
        "final_unmatched_epc_rows": int(len(final_unmatched)),
        "nearest_thresholds_m": NEAREST_THRESHOLDS_M,
        "dbt_crs": str(dbt.crs),
        "lat_col": lat_col,
        "lon_col": lon_col,
        "building_id_col": building_id_col
    }

    with open(os.path.join(out_dir, "join_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # Save raw match tables
    matched_all.to_csv(os.path.join(out_dir, "epc_matched_all_rows.csv"), index=False)
    pd.DataFrame(final_unmatched).to_csv(os.path.join(out_dir, "epc_unmatched_rows.csv"), index=False)

    print("\n✅ Saved match tables and summary.")

    # -----------------------------
    # Best EPC per building
    # -----------------------------
    print("\n--- Step 4: Select best EPC per building ---")
    best = best_epc_per_building(matched_all, building_id_col)
    best.to_csv(os.path.join(out_dir, "epc_best_per_building_v2.csv"), index=False)

    # Count EPCs per building
    cnt = matched_all.groupby(building_id_col).size().reset_index(name="epc_rows_matched_to_building")
    cnt.to_csv(os.path.join(out_dir, "epc_count_per_building.csv"), index=False)

    print(f"✅ Best EPC per building saved: {len(best)} buildings labeled now (upper bound).")

    # -----------------------------
    # Optional: create buildings_with_epc_v2 (DBT buildings + EPC best)
    # -----------------------------
    print("\n--- Step 5: Create buildings_with_epc_v2 (DBT + best EPC) ---")
    dbt_tab = dbt.drop(columns=["geometry"]).copy()
    buildings_with_epc = dbt_tab.merge(best.drop(columns=["geometry"], errors="ignore"), on=building_id_col, how="left")

    buildings_with_epc.to_csv(os.path.join(out_dir, "buildings_with_epc_v2.csv"), index=False)
    print("✅ Saved buildings_with_epc_v2.csv")

    print("\n============================================================")
    print("RECOVER SPATIAL JOIN v2 – COMPLETE")
    print("Output folder:")
    print(out_dir)
    print("============================================================")


def fiona_listlayers(gpkg_path: str):
    # Lazy import to avoid hard dependency unless needed
    try:
        import fiona
        return list(fiona.listlayers(gpkg_path))
    except Exception:
        return []


if __name__ == "__main__":
    main()