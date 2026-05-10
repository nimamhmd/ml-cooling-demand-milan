import os
import glob
import pandas as pd
import geopandas as gpd

# ============================================================
# CONFIGURATION
# ============================================================

CENED_ROOT = r"P:\Nima\23-11-2025\Data Collections\CENED"
BATCH_DIR = os.path.join(CENED_ROOT, "geo_outputs")
OUT_DIR = os.path.join(CENED_ROOT, "merged_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Output files
OUT_GPKG = os.path.join(OUT_DIR, "epc_geocoded_full.gpkg")       # best for GIS
OUT_GEOJSON = os.path.join(OUT_DIR, "epc_geocoded_full.geojson") # optional (big)
OUT_PARQUET = os.path.join(OUT_DIR, "epc_geocoded_full.parquet") # best for Python

# ============================================================
# 1) FIND BATCH FILES
# ============================================================

batch_files = sorted(glob.glob(os.path.join(BATCH_DIR, "epc_geocoded_batch_*.geojson")))

print("============================================================")
print("EPC BATCH MERGE")
print("============================================================")
print(f"Batch folder: {BATCH_DIR}")
print(f"Output folder: {OUT_DIR}")
print(f"Number of batch files found: {len(batch_files)}")

if len(batch_files) == 0:
    raise FileNotFoundError(
        "No batch files found. Check that your files are in:\n"
        f"{BATCH_DIR}\n"
        "and named like epc_geocoded_batch_XXXXX.geojson"
    )

print("First 3 files:", batch_files[:3])
print("Last 3 files:", batch_files[-3:])
print("============================================================\n")

# ============================================================
# 2) LOAD + CONCATENATE
# ============================================================

gdfs = []
for i, fp in enumerate(batch_files, start=1):
    g = gpd.read_file(fp)

    # Ensure CRS is EPSG:4326
    if g.crs is None:
        g = g.set_crs("EPSG:4326")
    else:
        g = g.to_crs("EPSG:4326")

    # Drop null/empty geometries right away
    g = g[~g.geometry.isna()]
    g = g[~g.geometry.is_empty]

    gdfs.append(g)

    if i % 25 == 0 or i == len(batch_files):
        print(f"Loaded {i}/{len(batch_files)} files...")

epc = pd.concat(gdfs, ignore_index=True)
epc = gpd.GeoDataFrame(epc, geometry="geometry", crs="EPSG:4326")

print("\nMerged EPC rows (before dedupe):", len(epc))

# ============================================================
# 3) BASIC CHECKS
# ============================================================

print("Null geometries:", epc.geometry.isna().sum())
print("Empty geometries:", epc.geometry.is_empty.sum())

# Add coordinate columns for sanity checks / dedupe fallback
epc["lon"] = epc.geometry.x
epc["lat"] = epc.geometry.y

print("Longitude range:", (float(epc["lon"].min()), float(epc["lon"].max())))
print("Latitude range:", (float(epc["lat"].min()), float(epc["lat"].max())))

# ============================================================
# 4) DEDUPLICATION (IMPORTANT)
# ============================================================

candidate_id_cols = [
    "CODICE_CERTIFICATO", "COD_APE", "ID_CERTIFICATO", "ID_APE", "APE_ID",
    "ID", "ID_CERT", "CODICE", "CODICEAPE", "NUM_CERT", "CODICE_SACE"
]

id_col = next((c for c in candidate_id_cols if c in epc.columns), None)
print("\nDetected ID column for dedupe:", id_col)

before = len(epc)

if id_col:
    epc = epc.drop_duplicates(subset=[id_col])
else:
    # fallback based on address + coords (or coords only)
    addr_col = None
    if "INDIRIZZO" in epc.columns:
        addr_col = "INDIRIZZO"
    elif "full_address" in epc.columns:
        addr_col = "full_address"

    if addr_col:
        epc = epc.drop_duplicates(subset=[addr_col, "lat", "lon"])
        print(f"Fallback dedupe used: {addr_col} + lat/lon")
    else:
        epc = epc.drop_duplicates(subset=["lat", "lon"])
        print("Fallback dedupe used: lat/lon only")

after = len(epc)
print(f"Deduped EPC rows: {before} -> {after}")

# ============================================================
# 5) SAVE OUTPUTS
# ============================================================

print("\nSaving outputs...")

# (A) GeoPackage
epc.to_file(OUT_GPKG, layer="epc", driver="GPKG")
print("Saved GeoPackage:", OUT_GPKG)

# (B) Parquet (fast for Python)
epc.to_parquet(OUT_PARQUET, index=False)
print("Saved Parquet:", OUT_PARQUET)

# (C) GeoJSON (optional, can be very large)
# Uncomment if you really need GeoJSON as a single file.
# epc.to_file(OUT_GEOJSON, driver="GeoJSON")
# print("Saved GeoJSON:", OUT_GEOJSON)

print("\n✅ DONE. Master EPC dataset created.")
print("Next step: spatial join EPC points to building footprints.")
