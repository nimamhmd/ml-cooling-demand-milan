import os
import pandas as pd
import geopandas as gpd

# =========================
# CONFIG (EDIT THESE PATHS)
# =========================

# EPC merged output (from your CENED merge script)
EPC_GPKG = r"P:\Nima\23-11-2025\Data Collections\CENED\merged_outputs\epc_geocoded_full.gpkg"

# Buildings polygons (from ClimateProcessingProject Stage 9)
BUILDINGS_GEOJSON = r"P:\Nima\23-11-2025\PythonProjects\ClimateProcessingProject\data\Buildings\buildings_geom_features.geojson"

# Output folder (choose where you want results)
OUT_DIR = r"P:\Nima\23-11-2025\Data Collections\CENED\spatial_join_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_JOIN_GPKG = os.path.join(OUT_DIR, "epc_joined_to_buildings.gpkg")
OUT_JOIN_CSV  = os.path.join(OUT_DIR, "epc_joined_to_buildings.csv")

# Optional: nearest join settings (meters)
DO_NEAREST_FOR_UNMATCHED = True
MAX_NEAREST_DISTANCE_M = 50   # 30–80 m typical. Start with 50m.

# =========================
# LOAD DATA
# =========================
print("\n[1] Loading EPC points...")
epc = gpd.read_file(EPC_GPKG)
print(f"   EPC rows: {len(epc):,}")
print(f"   EPC CRS : {epc.crs}")

print("\n[2] Loading Buildings polygons...")
bld = gpd.read_file(BUILDINGS_GEOJSON)
print(f"   Buildings rows: {len(bld):,}")
print(f"   Buildings CRS : {bld.crs}")

# =========================
# CRS HARMONIZATION
# =========================
print("\n[3] Reprojecting both to EPSG:4326 (WGS84)...")
epc = epc.to_crs("EPSG:4326")
bld = bld.to_crs("EPSG:4326")

# Basic geometry sanity checks
epc = epc[~epc.geometry.is_empty & epc.geometry.notna()].copy()
bld = bld[~bld.geometry.is_empty & bld.geometry.notna()].copy()

print(f"   EPC CRS after fix: {epc.crs}")
print(f"   BLD CRS after fix: {bld.crs}")

# =========================
# MAIN SPATIAL JOIN: WITHIN
# =========================
print("\n[4] Spatial join: EPC point WITHIN Building polygon...")
# Keep only necessary building identifiers + geometry (but keep your geom features if you want)
# Here I keep: building_id (if exists), FEATURE_ID, EDIFC_ID etc. Adjust if needed.
bld_cols = [c for c in bld.columns if c != "geometry"]
# You can reduce columns later for ML, but for traceability keep them now.

joined_within = gpd.sjoin(
    epc,
    bld,
    how="left",
    predicate="within"
)

matched_count = joined_within["index_right"].notna().sum()
unmatched_count = joined_within["index_right"].isna().sum()

print(f"   Matched (within): {matched_count:,}")
print(f"   Unmatched        : {unmatched_count:,}")

# =========================
# OPTIONAL: NEAREST FOR UNMATCHED
# =========================
if DO_NEAREST_FOR_UNMATCHED and unmatched_count > 0:
    print("\n[5] Nearest-join for unmatched points (optional)...")

    # Work in a metric CRS for distance calculations
    # EPSG:32632 = UTM zone 32N (works for Milan)
    epc_m = epc.to_crs("EPSG:32632")
    bld_m = bld.to_crs("EPSG:32632")

    # Identify unmatched points based on within-join result
    unmatched_idx = joined_within[joined_within["index_right"].isna()].index
    epc_unmatched = epc_m.loc[unmatched_idx].copy()

    print(f"   Unmatched points to try nearest: {len(epc_unmatched):,}")

    # nearest join gives nearest polygon and distance
    nearest = gpd.sjoin_nearest(
        epc_unmatched,
        bld_m,
        how="left",
        distance_col="nearest_dist_m"
    )

    # Filter by max distance threshold
    nearest_ok = nearest[nearest["nearest_dist_m"] <= MAX_NEAREST_DISTANCE_M].copy()
    print(f"   Nearest matched within {MAX_NEAREST_DISTANCE_M} m: {len(nearest_ok):,}")

    # Convert nearest_ok back to EPSG:4326 to merge into main result
    nearest_ok = nearest_ok.to_crs("EPSG:4326")

    # Update the original joined_within for these indices
    # We replace the rows in joined_within at nearest_ok.index with nearest_ok values (building columns + index_right)
    # First ensure column alignment
    for col in nearest_ok.columns:
        if col not in joined_within.columns:
            joined_within[col] = pd.NA

    joined_within.loc[nearest_ok.index, nearest_ok.columns] = nearest_ok[nearest_ok.columns]

    # Recompute counts after nearest
    matched_after = joined_within["index_right"].notna().sum()
    unmatched_after = joined_within["index_right"].isna().sum()
    print(f"   Matched after nearest: {matched_after:,}")
    print(f"   Still unmatched      : {unmatched_after:,}")

# =========================
# CLEAN OUTPUT + SAVE
# =========================
print("\n[6] Preparing outputs...")

# Drop sjoin helper columns if you want (keep them if you need traceability)
# joined_within has index_right
# keep it for auditing
out_gdf = joined_within.copy()

# Save GPKG (recommended)
print(f"\n[7] Saving GeoPackage:\n   {OUT_JOIN_GPKG}")
out_gdf.to_file(OUT_JOIN_GPKG, layer="epc_joined", driver="GPKG")

# Save CSV (geometry removed)
print(f"\n[8] Saving CSV:\n   {OUT_JOIN_CSV}")
out_df = pd.DataFrame(out_gdf.drop(columns="geometry"))
out_df.to_csv(OUT_JOIN_CSV, index=False)

print("\n✅ DONE: EPC points spatially joined to buildings.")
print("Next step: aggregate EPC to building-level (one row per building).")
