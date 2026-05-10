import os
import pandas as pd
import geopandas as gpd

# =========================
# PATHS (EDIT ONLY IF NEEDED)
# =========================
BUILDINGS_GEOJSON = r"P:\Nima\23-11-2025\PythonProjects\ClimateProcessingProject\data\Buildings\buildings_geom_features.geojson"
EPC_BEST_PARQ     = r"P:\Nima\23-11-2025\Data Collections\CENED\spatial_join_outputs\epc_best_per_building.parquet"

OUT_DIR = r"P:\Nima\23-11-2025\Data Collections\CENED\building_level_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_GPKG   = os.path.join(OUT_DIR, "buildings_with_epc.gpkg")
OUT_PARQ   = os.path.join(OUT_DIR, "buildings_with_epc.parquet")
OUT_CSV    = os.path.join(OUT_DIR, "buildings_with_epc.csv")

# =========================
# LOAD
# =========================
print("\n[1] Loading buildings...")
bld = gpd.read_file(BUILDINGS_GEOJSON)
bld = bld.to_crs("EPSG:4326")
print("   Buildings rows:", len(bld))
print("   Buildings CRS :", bld.crs)

print("\n[2] Loading EPC (one row per building)...")
epc = pd.read_parquet(EPC_BEST_PARQ)
print("   EPC rows:", len(epc))
print("   EPC columns:", len(epc.columns))

# =========================
# DETECT BUILDING KEY
# =========================
building_candidates = ["building_id", "EDIFC_ID", "FEATURE_ID", "ID_ZRIL"]
bld_key = next((c for c in building_candidates if c in bld.columns), None)
epc_key = next((c for c in building_candidates if c in epc.columns), None)

print("\n[3] Detected join keys:")
print("   Building key in buildings:", bld_key)
print("   Building key in epc      :", epc_key)

if bld_key is None or epc_key is None:
    raise ValueError("Building join key not found in one of the datasets. Check column names.")

# Ensure both keys are string (avoid join failure)
bld[bld_key] = bld[bld_key].astype(str)
epc[epc_key] = epc[epc_key].astype(str)

# =========================
# MERGE (LEFT JOIN: keep all buildings)
# =========================
print("\n[4] Merging EPC into buildings (left join)...")
merged = bld.merge(epc, left_on=bld_key, right_on=epc_key, how="left")

# Quick stats
matched = merged[epc_key].notna().sum()
print("   Rows after merge:", len(merged))
print("   Buildings with EPC matched:", matched)
print("   Buildings without EPC:", len(merged) - matched)

# =========================
# SAVE OUTPUTS
# =========================
print("\n[5] Saving outputs...")

# Save GPKG
merged.to_file(OUT_GPKG, layer="buildings_epc", driver="GPKG")
print("   Saved:", OUT_GPKG)

# Save Parquet (drop geometry -> WKB handled by geopandas, but keep in GeoParquet is not always consistent)
# Best practice: save a non-geo parquet for ML + keep geo in gpkg
df_out = pd.DataFrame(merged.drop(columns="geometry"))
df_out.to_parquet(OUT_PARQ, index=False)
print("   Saved:", OUT_PARQ)

# Save CSV for inspection
df_out.to_csv(OUT_CSV, index=False)
print("   Saved:", OUT_CSV)

print("\n✅ DONE: buildings_with_epc created.")
print("Next step: attach climate climatology (2005–2020) to create ML-ready dataset.")
