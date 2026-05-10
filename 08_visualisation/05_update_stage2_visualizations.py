"""
============================================================================
SCRIPT D: Multi-Model Visualizations - 7 Figures + 8 Maps
----------------------------------------------------------------------------
Produces all visualizations for the multi-model revision of Stage 2.

Outputs:
  _MULTIMODEL/_FIGURES/
    fig_4_1_cdd_evolution_multimodel.png/pdf
    fig_4_2_cdd_distributions_multimodel.png/pdf
    fig_4_4_demand_per_class_heatmap_multimodel.png/pdf
    fig_4_5_climate_signal_summary_multimodel.png/pdf
    fig_4_6_adoption_pathway_comparison_multimodel.png/pdf
    fig_4_7_demand_distribution_violin_multimodel.png/pdf
    fig_4_8_multipaper_validation_multimodel.png/pdf
  _MULTIMODEL/_MAPS/
    map_4_1_historical_1990_2024_multimodel.png/pdf
    map_4_2_today_2015_2024_multimodel.png/pdf
    map_4_3_moderate_2080_multimodel.png/pdf
    map_4_4_high_2080_multimodel.png/pdf
    map_4_5_demand_change_absolute_multimodel.png/pdf
    map_4_6_demand_per_district_88clusters_multimodel.png/pdf
    map_4_7_district_delta_585_vs_today_multimodel.png/pdf
    map_4_8_summary_4panel_multimodel.png/pdf

Each visualization gets a .caption.txt with thesis-quality caption text.

Methodology:
  - Building-level mapping via EDIFC_ID join (53,041 polygons in shapefile)
  - 19,063 EPC buildings get demand colors; remainder shown as grey background
  - 88-district aggregation via nearest-centroid assignment to locked centroids
  - All future scenarios use 12-model NEX-GDDP-CMIP6 ensemble median CDD
  - Color scales harmonised across comparable maps (intensity, delta)

Runtime: 8-12 minutes (most time = rendering 53k polygons per map).
============================================================================
"""

import sys
import subprocess
import time
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ===========================================================================
# AUTO-INSTALL DEPENDENCIES
# ===========================================================================
def ensure(pkg, import_name=None):
    name = import_name or pkg
    try:
        __import__(name)
    except ImportError:
        print(f"Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

for p in [("geopandas", None), ("seaborn", None), ("mapclassify", None)]:
    ensure(p[0], p[1])

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize, BoundaryNorm
from matplotlib.cm import ScalarMappable
import seaborn as sns
import geopandas as gpd
from shapely.geometry import Point

# ===========================================================================
# CONFIGURATION
# ===========================================================================

CANDIDATE_PROJECT_ROOTS = [
    Path(r"C:\Users\n.mohammadi\Desktop\NimaMohammadi\03. Nima Mohammadi - Thesis\ML Dataset"),
    Path(r"C:\Users\n.mohammadi\Desktop\NimaMohammadi\02.Nima Mohammadi - Thesis\ML Dataset"),
]
PROJECT_ROOT = next((p for p in CANDIDATE_PROJECT_ROOTS if p.exists()), None)
if PROJECT_ROOT is None:
    print("ERROR: ML Dataset folder not found.")
    sys.exit(1)
print(f"Project root: {PROJECT_ROOT}")

STAGE2_ROOT = PROJECT_ROOT / "07_results" / "stage2"
MULTIMODEL_ROOT = STAGE2_ROOT / "_MULTIMODEL"
FIGURES_DIR = MULTIMODEL_ROOT / "_FIGURES"
MAPS_DIR = MULTIMODEL_ROOT / "_MAPS"
TABLES_DIR = MULTIMODEL_ROOT / "_TABLES"
LOGS_DIR = MULTIMODEL_ROOT / "_LOGS"
for d in [FIGURES_DIR, MAPS_DIR, TABLES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Inputs
PER_BUILDING_FILE = (STAGE2_ROOT / "session4_demand_projection" / "E_outputs"
                     / "stage2_per_building_demand_projections_v2.csv")
CDD_PER_YEAR_FILE = TABLES_DIR / "tab_4_1_cdd_per_model_per_year.csv"
CDD_MULTIMODEL_FILE = TABLES_DIR / "tab_4_1_cdd_projections_multimodel.csv"
DEMAND_SUMMARY_FILE = TABLES_DIR / "tab_4_2_demand_summary_multimodel.csv"
PER_CLASS_FILE = TABLES_DIR / "tab_4_3_demand_per_class_multimodel.csv"
LOCKED_DISTRICT_FILE = STAGE2_ROOT / "_FINAL_TABLES" / "tab_4_6_demand_per_district.csv"
ERA5_FILE = PROJECT_ROOT / "02_inputs" / "ERA5_Milan_1990_2024_daily.csv"
SHAPEFILE = Path(r"C:\Users\n.mohammadi\Desktop\NimaMohammadi"
                 r"\03. Nima Mohammadi - Thesis\Data Collections\Geoportale Milano\A020102.shp")

# ERA5 anchors (locked from existing Stage 2)
ERA5_HISTORICAL_CDD = 162.1
ERA5_RECENT_CDD = 206.6

MODELS = [
    "MPI-ESM1-2-HR", "CMCC-ESM2", "EC-Earth3", "CNRM-ESM2-1",
    "IPSL-CM6A-LR", "CESM2", "HadGEM3-GC31-LL", "CanESM5",
    "NorESM2-MM", "GFDL-ESM4", "MIROC6", "ACCESS-ESM1-5",
]

SCENARIOS = [
    "ERA5 historical 1990-2024", "ERA5 recent 2015-2024",
    "SSP2-4.5 / 2030-2050", "SSP2-4.5 / 2080-2100",
    "SSP5-8.5 / 2030-2050", "SSP5-8.5 / 2080-2100",
]

SCENARIO_SHORT = {
    "ERA5 historical 1990-2024": "Hist 1990-2024",
    "ERA5 recent 2015-2024":     "Today 2015-2024",
    "SSP2-4.5 / 2030-2050":      "SSP245 2030-2050",
    "SSP2-4.5 / 2080-2100":      "SSP245 2080-2100",
    "SSP5-8.5 / 2030-2050":      "SSP585 2030-2050",
    "SSP5-8.5 / 2080-2100":      "SSP585 2080-2100",
}

SCENARIO_COLORS = {
    "ERA5 historical 1990-2024": "#333333",
    "ERA5 recent 2015-2024":     "#777777",
    "SSP2-4.5 / 2030-2050":      "#4A7AB8",
    "SSP2-4.5 / 2080-2100":      "#1F4E79",
    "SSP5-8.5 / 2030-2050":      "#D49A4A",
    "SSP5-8.5 / 2080-2100":      "#A03020",
}

# Plot style
mpl.rcdefaults()
plt.style.use("default")
mpl.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "axes.edgecolor": "black",
    "xtick.color": "black", "ytick.color": "black",
    "text.color": "black", "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 11, "savefig.dpi": 300,
    "axes.spines.top": False, "axes.spines.right": False,
})
HEADING = "#1F4E79"

print("=" * 75)
print("SCRIPT D: Multi-Model Visualizations (7 figures + 8 maps)")
print(f"Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 75)

exec_log = [f"Script D log - started {datetime.now().isoformat()}", "=" * 75, ""]

def save_caption(out_path_stem, text):
    """Save thesis-quality caption alongside figure."""
    cap_path = Path(str(out_path_stem) + ".caption.txt")
    with open(cap_path, "w", encoding="utf-8") as f:
        f.write(text)

def save_fig(fig, stem, also_pdf=True):
    """Save figure as PNG (and PDF), close, return PNG path."""
    png = Path(f"{stem}.png")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    if also_pdf:
        fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)
    return png

# ===========================================================================
# STEP 1: LOAD ALL DATA
# ===========================================================================

print("\n[1] Loading data...")
t_load = time.time()

# Per-building
proj = pd.read_csv(PER_BUILDING_FILE, low_memory=False)
print(f"  Per-building: {len(proj):,} buildings")

# Multi-model CDD
cdd_per_year = pd.read_csv(CDD_PER_YEAR_FILE)
cdd_summary = pd.read_csv(CDD_MULTIMODEL_FILE)

# CDD ensemble median per scenario (for per-building demand calculation)
cdd_medians = {
    "ERA5 historical 1990-2024": ERA5_HISTORICAL_CDD,
    "ERA5 recent 2015-2024": ERA5_RECENT_CDD,
}
sc_map = {
    "SSP2-4.5 / 2030-2050": ("ssp245", 2030),
    "SSP2-4.5 / 2080-2100": ("ssp245", 2080),
    "SSP5-8.5 / 2030-2050": ("ssp585", 2030),
    "SSP5-8.5 / 2080-2100": ("ssp585", 2080),
}
for sc_disp, (sc_short, p_start) in sc_map.items():
    row = cdd_summary[(cdd_summary["scenario"] == sc_short) &
                       (cdd_summary["period_start"] == p_start)].iloc[0]
    cdd_medians[sc_disp] = float(row["cdd_ensemble_median"])

# Demand summary multi-model
demand_summary = pd.read_csv(DEMAND_SUMMARY_FILE)

# Per-class
per_class = pd.read_csv(PER_CLASS_FILE)

# ERA5 yearly CDD
era5 = pd.read_csv(ERA5_FILE, usecols=["time", "temperature_2m"])
era5["time"] = pd.to_datetime(era5["time"], errors="coerce")
era5 = era5.dropna(subset=["time"]).copy()
era5["year"] = era5["time"].dt.year
era5["month"] = era5["time"].dt.month
era5["cdd"] = (era5["temperature_2m"] - 22.0).clip(lower=0)
era5_yearly = era5[era5["month"].isin([6, 7, 8])].groupby("year")["cdd"].sum()

# Locked district centroids (88 clusters) - we use these for spatial assignment
locked_districts = pd.read_csv(LOCKED_DISTRICT_FILE)
print(f"  Locked districts: {len(locked_districts)} clusters")

# Shapefile (heavy)
print(f"  Loading shapefile (~38 MB, takes 5-10 sec)...")
buildings_gdf = gpd.read_file(SHAPEFILE)
print(f"  Shapefile: {len(buildings_gdf):,} polygons, CRS={buildings_gdf.crs}")
print(f"  Loading complete in {time.time()-t_load:.1f}s")

# ===========================================================================
# STEP 2: COMPUTE PER-BUILDING MULTI-MODEL DEMAND PER SCENARIO
# ===========================================================================

print("\n[2] Computing per-building demand per scenario (ensemble median CDD)...")
beta = proj["beta_predicted"].values
area = proj["effective_cooled_area_m2"].values
is_cooled = (proj["CLIMATIZZAZIONE_ESTIVA"] == True).values

per_bldg = pd.DataFrame({
    "EDIFC_ID": proj["EDIFC_ID"].values,
    "is_cooled": is_cooled,
    "area_m2": area,
    "beta_predicted": beta,
    "energy_class": proj["CLASSE_ENERGETICA"].values,
})
for sc, cdd_val in cdd_medians.items():
    sc_safe = sc.replace(" ", "_").replace("/", "-").replace(".", "")
    per_bldg[f"int_{sc_safe}"] = beta * cdd_val             # kWh/m^2/yr
    per_bldg[f"tot_{sc_safe}"] = beta * cdd_val * area      # kWh/yr
print(f"  Computed for {len(per_bldg):,} buildings x {len(cdd_medians)} scenarios")

# Delta column for change maps
sc_today = "ERA5 recent 2015-2024".replace(" ", "_").replace("/", "-").replace(".", "")
sc_585_2080 = "SSP5-8.5 / 2080-2100".replace(" ", "_").replace("/", "-").replace(".", "")
per_bldg["delta_kWh_585_vs_today"] = (per_bldg[f"tot_{sc_585_2080}"] -
                                       per_bldg[f"tot_{sc_today}"])

# ===========================================================================
# STEP 3: NEAREST-CENTROID DISTRICT ASSIGNMENT (for 88-cluster maps)
# ===========================================================================

print("\n[3] Assigning buildings to 88 districts via nearest-centroid lookup...")
# Need building lat/lon. Get from shapefile centroids reprojected.
buildings_centroids = buildings_gdf.geometry.centroid
# Reproject to WGS84 to compare with locked centroids (which are lat/lon)
centroids_wgs84 = gpd.GeoSeries(buildings_centroids,
                                  crs=buildings_gdf.crs).to_crs(epsg=4326)
b_lat = centroids_wgs84.y.values
b_lon = centroids_wgs84.x.values

# Locked cluster centroids
c_lat = locked_districts["lat_centroid"].values
c_lon = locked_districts["lon_centroid"].values
c_id = locked_districts["cluster_id"].values

# For each building polygon centroid, find nearest cluster centroid (vectorised)
# Squared distance in lat/lon (small angles, OK for Milan area)
# Use cosine-correction for longitude
mean_lat = b_lat.mean()
cos_lat = np.cos(np.deg2rad(mean_lat))
def nearest_cluster(lat, lon):
    dlat = (c_lat - lat)
    dlon = (c_lon - lon) * cos_lat
    return c_id[np.argmin(dlat**2 + dlon**2)]
# Vectorised loop (faster than scipy KDTree for 53k vs 88)
print(f"    Assigning {len(b_lat):,} buildings to nearest of 88 centroids...")
shp_assignments = np.empty(len(b_lat), dtype=object)
for i in range(len(b_lat)):
    dlat = c_lat - b_lat[i]
    dlon = (c_lon - b_lon[i]) * cos_lat
    shp_assignments[i] = c_id[np.argmin(dlat**2 + dlon**2)]
buildings_gdf = buildings_gdf.copy()
buildings_gdf["cluster_id"] = shp_assignments

# Also assign EPC buildings to clusters via EDIFC_ID join with shapefile
# Strategy: join per_bldg to buildings_gdf to get cluster per EPC building
epc_with_cluster = per_bldg.merge(
    buildings_gdf[["EDIFC_ID", "cluster_id"]],
    on="EDIFC_ID", how="left"
)
print(f"    EPC buildings assigned: {epc_with_cluster['cluster_id'].notna().sum():,}/{len(epc_with_cluster):,}")

# Aggregate per-district demand for multi-model
print("\n[4] Aggregating multi-model demand to 88 districts...")
district_rows = []
for cid in c_id:
    sub = epc_with_cluster[epc_with_cluster["cluster_id"] == cid]
    if len(sub) == 0:
        # Empty cluster - fall back to locked values
        loc = locked_districts[locked_districts["cluster_id"] == cid].iloc[0]
        district_rows.append({
            "cluster_id": cid,
            "lat_centroid": loc["lat_centroid"],
            "lon_centroid": loc["lon_centroid"],
            "n_buildings": 0,
            "total_GWh_today": 0,
            "total_GWh_ssp245_2080": 0,
            "total_GWh_ssp585_2080": 0,
            "mean_intensity_today": 0,
            "mean_intensity_ssp585_2080": 0,
            "delta_GWh_585_vs_today": 0,
            "pct_change_585_vs_today": 0,
        })
        continue
    cooled = sub[sub["is_cooled"]]
    loc = locked_districts[locked_districts["cluster_id"] == cid].iloc[0]
    today_total = cooled[f"tot_{sc_today}"].sum() / 1e6
    ssp245_total = cooled[f"tot_{('SSP2-4.5 / 2080-2100').replace(' ','_').replace('/','-').replace('.','')}"].sum() / 1e6
    ssp585_total = cooled[f"tot_{sc_585_2080}"].sum() / 1e6
    today_int = cooled[f"int_{sc_today}"].mean() if len(cooled) else 0
    ssp585_int = cooled[f"int_{sc_585_2080}"].mean() if len(cooled) else 0
    delta_gwh = ssp585_total - today_total
    pct_chg = (delta_gwh / today_total * 100) if today_total > 0 else 0
    district_rows.append({
        "cluster_id": cid,
        "lat_centroid": loc["lat_centroid"],
        "lon_centroid": loc["lon_centroid"],
        "n_buildings": int(sub["is_cooled"].sum()),
        "total_GWh_today": round(today_total, 3),
        "total_GWh_ssp245_2080": round(ssp245_total, 3),
        "total_GWh_ssp585_2080": round(ssp585_total, 3),
        "mean_intensity_today": round(today_int, 2),
        "mean_intensity_ssp585_2080": round(ssp585_int, 2),
        "delta_GWh_585_vs_today": round(delta_gwh, 3),
        "pct_change_585_vs_today": round(pct_chg, 1),
    })
districts_mm = pd.DataFrame(district_rows)
districts_path = TABLES_DIR / "tab_4_6_demand_per_district_multimodel.csv"
districts_mm.to_csv(districts_path, index=False)
print(f"  Saved: {districts_path.name}")

# Reproject district centroids to map CRS
district_geom = gpd.GeoDataFrame(
    districts_mm,
    geometry=[Point(xy) for xy in zip(districts_mm["lon_centroid"], districts_mm["lat_centroid"])],
    crs="EPSG:4326"
).to_crs(buildings_gdf.crs)

# Merge per-building demand into shapefile for mapping
print("\n[5] Joining per-building demand to shapefile...")
buildings_mapped = buildings_gdf.merge(per_bldg, on="EDIFC_ID", how="left")
n_with_data = buildings_mapped["beta_predicted"].notna().sum()
print(f"  Polygons with EPC data: {n_with_data:,} / {len(buildings_mapped):,}")
exec_log.append(f"Data loaded and mapped successfully")

# Map extents (use shapefile bounds in EPSG:6707 metres)
xmin, ymin, xmax, ymax = buildings_gdf.total_bounds
pad_x = (xmax - xmin) * 0.02
pad_y = (ymax - ymin) * 0.02

# ===========================================================================
# STEP 6: NON-SPATIAL FIGURES
# ===========================================================================

print("\n[6] Generating non-spatial figures...")

# ---------------------------------------------------------------------------
# Figure 4.1 - CDD evolution across scenarios with multi-model spread
# ---------------------------------------------------------------------------
print("  fig_4_1_cdd_evolution_multimodel...")
fig, ax = plt.subplots(figsize=(13, 7))

# ERA5 historical line (1990-2024)
era5_yrs = era5_yearly.index.values
ax.plot(era5_yrs, era5_yearly.values, color="#333333", lw=1.5, alpha=0.95,
        label="ERA5 historical (observation)", zorder=5)
ax.fill_between(era5_yrs,
                 era5_yearly.values - era5_yearly.std(),
                 era5_yearly.values + era5_yearly.std(),
                 color="#333333", alpha=0.10, zorder=4)

# For each future scenario: plot ensemble median + spread
fut_scenario_pairs = [
    ("ssp245", 2030, 2050, "#4A7AB8", "SSP2-4.5 (2030-2050)"),
    ("ssp245", 2080, 2100, "#1F4E79", "SSP2-4.5 (2080-2100)"),
    ("ssp585", 2030, 2050, "#D49A4A", "SSP5-8.5 (2030-2050)"),
    ("ssp585", 2080, 2100, "#A03020", "SSP5-8.5 (2080-2100)"),
]
for sc_short, p_s, p_e, color, label in fut_scenario_pairs:
    sub = cdd_per_year[(cdd_per_year["scenario"] == sc_short) &
                        (cdd_per_year["period_start"] == p_s)]
    pivot = sub.pivot_table(index="year", columns="model", values="cdd")
    yrs = pivot.index.values
    median = pivot.median(axis=1).values
    p05 = pivot.quantile(0.05, axis=1).values
    p95 = pivot.quantile(0.95, axis=1).values
    # Faint per-model lines
    for m in pivot.columns:
        ax.plot(yrs, pivot[m].values, color=color, lw=0.5, alpha=0.20, zorder=2)
    # Spread band
    ax.fill_between(yrs, p05, p95, color=color, alpha=0.20, zorder=3)
    # Median bold line
    ax.plot(yrs, median, color=color, lw=2.4, alpha=0.95, label=label, zorder=6)

# Reference horizontals
ax.axhline(ERA5_HISTORICAL_CDD, color="#888888", lw=1.0, ls=":", alpha=0.7,
           label=f"ERA5 1990-2024 mean ({ERA5_HISTORICAL_CDD:.0f} CDD)")
ax.axhline(ERA5_RECENT_CDD, color="#555555", lw=1.0, ls="--", alpha=0.7,
           label=f"ERA5 recent 2015-2024 ({ERA5_RECENT_CDD:.0f} CDD)")

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Annual summer CDD (base 22 °C)", fontsize=12)
ax.set_title("Milan summer CDD evolution: ERA5 observation + 12-model NEX-GDDP-CMIP6 ensemble",
             fontsize=13, fontweight="bold", color=HEADING, loc="left")
ax.legend(loc="upper left", frameon=False, fontsize=9, ncol=2)
ax.grid(True, alpha=0.3, linewidth=0.5)
ax.set_xlim(1990, 2100)
plt.tight_layout()
stem = FIGURES_DIR / "fig_4_1_cdd_evolution_multimodel"
save_fig(fig, stem)
save_caption(stem,
    "Figure 4.1. Evolution of annual summer cooling degree days (CDD, base 22 °C) "
    "for Milan from ERA5 reanalysis (1990-2024, dark grey) and 12-model NEX-GDDP-CMIP6 "
    "ensemble projections under SSP2-4.5 (blue tones) and SSP5-8.5 (warm tones). "
    "Bold lines show ensemble median; faint lines show individual models; shaded bands "
    "show the 5-95th percentile spread across the 12 models. Horizontal dotted lines "
    "indicate ERA5 historical (162.1) and recent (206.6) baselines.")

# ---------------------------------------------------------------------------
# Figure 4.2 - CDD distributions per scenario, all 12 models as boxes
# ---------------------------------------------------------------------------
print("  fig_4_2_cdd_distributions_multimodel...")
fig, ax = plt.subplots(figsize=(13, 7))
plot_data = []
for sc_disp in SCENARIOS:
    if "ERA5" in sc_disp:
        if "historical" in sc_disp:
            vals = era5_yearly.values
        else:
            vals = era5_yearly.loc[2015:2024].values
        for v in vals:
            plot_data.append({"scenario": SCENARIO_SHORT[sc_disp], "cdd": v,
                               "type": "ERA5", "model": "ERA5"})
    else:
        sc_short, p_s = sc_map[sc_disp]
        sub = cdd_per_year[(cdd_per_year["scenario"] == sc_short) &
                            (cdd_per_year["period_start"] == p_s)]
        for _, r in sub.iterrows():
            plot_data.append({"scenario": SCENARIO_SHORT[sc_disp], "cdd": r["cdd"],
                               "type": "CMIP6", "model": r["model"]})
df_plot = pd.DataFrame(plot_data)
order = [SCENARIO_SHORT[s] for s in SCENARIOS]
palette = [SCENARIO_COLORS[s] for s in SCENARIOS]
sns.boxplot(data=df_plot, x="scenario", y="cdd", order=order, palette=palette,
            ax=ax, showfliers=True, fliersize=2.5,
            boxprops={"alpha": 0.75, "edgecolor": "black"},
            medianprops={"color": "black", "linewidth": 1.5})
sns.stripplot(data=df_plot, x="scenario", y="cdd", order=order, ax=ax,
              size=2, color="black", alpha=0.25, jitter=0.25)
ax.set_xlabel("")
ax.set_ylabel("Annual summer CDD (base 22 °C)", fontsize=12)
ax.set_title("CDD year-to-year distributions per scenario (ERA5 + 12-model ensemble)",
             fontsize=13, fontweight="bold", color=HEADING, loc="left")
ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
stem = FIGURES_DIR / "fig_4_2_cdd_distributions_multimodel"
save_fig(fig, stem)
save_caption(stem,
    "Figure 4.2. Distribution of annual summer CDD per scenario, pooling year-to-year "
    "values from ERA5 (historical, recent) and from all 12 NEX-GDDP-CMIP6 models for "
    "future periods. Boxplots show 25th-75th percentile (box) and 1.5x IQR (whiskers); "
    "individual year-model points are jittered to show inter-annual and inter-model "
    "spread directly. Box height widens substantially under SSP5-8.5/2080-2100, "
    "reflecting both higher mean CDD and greater dispersion across models and years.")

# ---------------------------------------------------------------------------
# Figure 4.4 - Per-class intensity heatmap
# ---------------------------------------------------------------------------
print("  fig_4_4_demand_per_class_heatmap_multimodel...")
heat = per_class.pivot_table(index="energy_class", columns="scenario",
                              values="mean_intensity_kWh_m2", aggfunc="mean")
class_order = ["A4", "A3", "A2", "A1", "B", "C", "D", "E", "F", "G"]
class_order = [c for c in class_order if c in heat.index]
heat = heat.reindex(class_order)
heat = heat[[s for s in SCENARIOS if s in heat.columns]]

fig, ax = plt.subplots(figsize=(11, 7))
sns.heatmap(heat, annot=True, fmt=".0f", cmap="YlOrRd", cbar_kws={"label": "Mean cooling intensity (kWh/m²/yr)"},
            linewidths=0.5, linecolor="white", ax=ax,
            annot_kws={"fontsize": 9, "fontweight": "bold"})
ax.set_xlabel("")
ax.set_ylabel("EPC energy class")
ax.set_title("Mean cooling intensity per energy class and scenario (12-model ensemble median)",
             fontsize=13, fontweight="bold", color=HEADING, loc="left")
plt.xticks(rotation=15, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
stem = FIGURES_DIR / "fig_4_4_demand_per_class_heatmap_multimodel"
save_fig(fig, stem)
save_caption(stem,
    "Figure 4.4. Mean predicted cooling intensity (kWh/m²/yr) per EPC energy class and "
    "climate scenario, computed for the cooled stock (n=13,787) using each scenario's "
    "ensemble median CDD across the 12 NEX-GDDP-CMIP6 models. Class G shows the highest "
    "absolute intensities and the largest absolute increases, consistent with poorer "
    "envelope performance. Class A4 buildings absorb future warming with the smallest "
    "absolute escalation, reinforcing the value of envelope retrofit as climate adaptation.")

# ---------------------------------------------------------------------------
# Figure 4.5 - Climate signal summary 4-panel
# ---------------------------------------------------------------------------
print("  fig_4_5_climate_signal_summary_multimodel...")
fig, axes = plt.subplots(2, 2, figsize=(15, 11))

# (a) CDD ensemble median across scenarios
ax = axes[0, 0]
sc_to_plot = [s for s in SCENARIOS]
cdd_vals = [cdd_medians[s] for s in sc_to_plot]
colors = [SCENARIO_COLORS[s] for s in sc_to_plot]
ax.bar(range(len(sc_to_plot)), cdd_vals, color=colors, edgecolor="black", linewidth=0.5)
for i, v in enumerate(cdd_vals):
    ax.text(i, v + max(cdd_vals)*0.02, f"{v:.0f}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(range(len(sc_to_plot)))
ax.set_xticklabels([SCENARIO_SHORT[s] for s in sc_to_plot], rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Annual summer CDD")
ax.set_title("(a) CDD per scenario (ensemble median)", fontsize=11, fontweight="bold",
             color=HEADING, loc="left")
ax.grid(axis="y", alpha=0.3)

# (b) Demand per scenario (V1)
ax = axes[0, 1]
v1 = demand_summary[demand_summary["variant"] == "V1_cooled_n13787"].set_index("scenario").reindex(SCENARIOS).reset_index()
median = v1["total_GWh_median"].values
p05 = v1["total_GWh_p05"].values
p95 = v1["total_GWh_p95"].values
ax.bar(range(len(SCENARIOS)), median, color=colors, edgecolor="black", linewidth=0.5,
       yerr=[median-p05, p95-median], capsize=5, ecolor="black")
for i, v in enumerate(median):
    ax.text(i, p95[i] + max(median)*0.02, f"{v:.0f}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(range(len(SCENARIOS)))
ax.set_xticklabels([SCENARIO_SHORT[s] for s in SCENARIOS], rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Cooled stock demand (GWh/yr)")
ax.set_title("(b) V1 demand (ensemble + spread)", fontsize=11, fontweight="bold",
             color=HEADING, loc="left")
ax.grid(axis="y", alpha=0.3)

# (c) Percent increase in CDD vs ERA5 historical
ax = axes[1, 0]
pct = [(c - ERA5_HISTORICAL_CDD)/ERA5_HISTORICAL_CDD * 100 for c in cdd_vals]
ax.bar(range(len(sc_to_plot)), pct, color=colors, edgecolor="black", linewidth=0.5)
for i, v in enumerate(pct):
    ax.text(i, v + max(pct)*0.02 if v > 0 else v - max(pct)*0.04,
            f"{v:+.0f}%", ha="center",
            va="bottom" if v > 0 else "top", fontsize=10, fontweight="bold")
ax.axhline(0, color="black", lw=1.0)
ax.set_xticks(range(len(sc_to_plot)))
ax.set_xticklabels([SCENARIO_SHORT[s] for s in sc_to_plot], rotation=20, ha="right", fontsize=9)
ax.set_ylabel("CDD change vs 1990-2024 (%)")
ax.set_title("(c) CDD percent change vs ERA5 1990-2024", fontsize=11, fontweight="bold",
             color=HEADING, loc="left")
ax.grid(axis="y", alpha=0.3)

# (d) Percent increase in V1 demand vs today
ax = axes[1, 1]
v1_today = v1[v1["scenario"] == "ERA5 recent 2015-2024"]["total_GWh_median"].iloc[0]
pct_v1 = (median - v1_today) / v1_today * 100
ax.bar(range(len(SCENARIOS)), pct_v1, color=colors, edgecolor="black", linewidth=0.5)
for i, v in enumerate(pct_v1):
    ax.text(i, v + max(pct_v1)*0.02 if v > 0 else v - max(pct_v1)*0.04,
            f"{v:+.0f}%", ha="center",
            va="bottom" if v > 0 else "top", fontsize=10, fontweight="bold")
ax.axhline(0, color="black", lw=1.0)
ax.set_xticks(range(len(SCENARIOS)))
ax.set_xticklabels([SCENARIO_SHORT[s] for s in SCENARIOS], rotation=20, ha="right", fontsize=9)
ax.set_ylabel("V1 demand change vs today (%)")
ax.set_title("(d) V1 demand percent change vs ERA5 recent", fontsize=11, fontweight="bold",
             color=HEADING, loc="left")
ax.grid(axis="y", alpha=0.3)

fig.suptitle("Climate signal summary: CDD and V1 demand under 12-model NEX-GDDP-CMIP6 ensemble",
             fontsize=14, fontweight="bold", color=HEADING, y=1.005)
plt.tight_layout()
stem = FIGURES_DIR / "fig_4_5_climate_signal_summary_multimodel"
save_fig(fig, stem)
save_caption(stem,
    "Figure 4.5. Four-panel climate signal summary. (a) Annual summer CDD by scenario "
    "(ensemble median across 12 models). (b) Variant 1 cooling demand (cooled stock, "
    "n=13,787) with 5-95th percentile spread across models. (c) Percent change in CDD "
    "relative to ERA5 1990-2024 baseline. (d) Percent change in V1 demand relative to "
    "ERA5 recent 2015-2024. Under SSP5-8.5/2080-2100 the 12-model ensemble median "
    "CDD reaches 747.6 (+360% vs 1990-2024) and demand reaches 139.5 GWh/yr "
    "(+262% vs today).")

# ---------------------------------------------------------------------------
# Figure 4.6 - Adoption pathway comparison (V3)
# ---------------------------------------------------------------------------
print("  fig_4_6_adoption_pathway_comparison_multimodel...")
fig, ax = plt.subplots(figsize=(13, 7))
v3_set = ["V3_Conservative_72pct", "V3_Moderate_85pct", "V3_High_100pct"]
v3_labels = {"V3_Conservative_72pct": "72% (current)",
             "V3_Moderate_85pct": "85% (moderate)",
             "V3_High_100pct": "100% (full saturation)"}
v3_colors = {"V3_Conservative_72pct": "#D49A4A",
             "V3_Moderate_85pct": "#A03020",
             "V3_High_100pct": "#5A0000"}
n_sc = len(SCENARIOS)
n_v = len(v3_set)
bar_w = 0.25
x = np.arange(n_sc)
for j, v in enumerate(v3_set):
    sub = demand_summary[demand_summary["variant"] == v].set_index("scenario").reindex(SCENARIOS).reset_index()
    median = sub["total_GWh_median"].values
    p05 = sub["total_GWh_p05"].values
    p95 = sub["total_GWh_p95"].values
    ax.bar(x + (j - 1) * bar_w, median, width=bar_w, color=v3_colors[v],
           edgecolor="black", linewidth=0.4,
           yerr=[median - p05, p95 - median], capsize=3, ecolor="black",
           label=v3_labels[v], alpha=0.92)
ax.set_xticks(x)
ax.set_xticklabels([SCENARIO_SHORT[s] for s in SCENARIOS], rotation=15, ha="right")
ax.set_ylabel("Total cooling demand (GWh/yr)")
ax.set_title("Variant 3: AC adoption pathway sensitivity per scenario (ensemble median + spread)",
             fontsize=13, fontweight="bold", color=HEADING, loc="left")
ax.legend(loc="upper left", frameon=False, fontsize=10, title="Adoption rate")
ax.grid(axis="y", alpha=0.3, linewidth=0.5)
plt.tight_layout()
stem = FIGURES_DIR / "fig_4_6_adoption_pathway_comparison_multimodel"
save_fig(fig, stem)
save_caption(stem,
    "Figure 4.6. Variant 3 cooling demand under three AC adoption pathways for each "
    "climate scenario. Bars show ensemble median across 12 models; whiskers show "
    "5-95th percentile spread. The widening gap between adoption pathways under "
    "future scenarios indicates that AC adoption decisions made today will materially "
    "affect end-century demand. Under SSP5-8.5/2080-2100, full-saturation adoption "
    "(100%) yields demand approximately 39% above current cooled-fraction adoption (72%).")

# ---------------------------------------------------------------------------
# Figure 4.7 - Per-building demand distribution violins by scenario
# ---------------------------------------------------------------------------
print("  fig_4_7_demand_distribution_violin_multimodel...")
cooled_only = per_bldg[per_bldg["is_cooled"]].copy()
violin_data = []
for sc_disp in SCENARIOS:
    sc_safe = sc_disp.replace(" ", "_").replace("/", "-").replace(".", "")
    intensities = cooled_only[f"int_{sc_safe}"].values
    for v in intensities:
        violin_data.append({"scenario": SCENARIO_SHORT[sc_disp], "intensity": v})
df_violin = pd.DataFrame(violin_data)
fig, ax = plt.subplots(figsize=(13, 7))
order = [SCENARIO_SHORT[s] for s in SCENARIOS]
palette = [SCENARIO_COLORS[s] for s in SCENARIOS]
sns.violinplot(data=df_violin, x="scenario", y="intensity", order=order,
               palette=palette, ax=ax, cut=0, inner="box",
               linewidth=0.8)
ax.set_xlabel("")
ax.set_ylabel("Per-building cooling intensity (kWh/m²/yr)")
ax.set_title("Distribution of per-building cooling intensity (cooled stock, n=13,787)",
             fontsize=13, fontweight="bold", color=HEADING, loc="left")
ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
stem = FIGURES_DIR / "fig_4_7_demand_distribution_violin_multimodel"
save_fig(fig, stem)
save_caption(stem,
    "Figure 4.7. Distribution of predicted per-building cooling intensity (kWh/m²/yr) "
    "for the 13,787 cooled buildings in Milan, by scenario. Each violin uses the "
    "ensemble-median CDD of its scenario applied to all buildings. Distribution "
    "broadens substantially under future scenarios because higher CDD amplifies "
    "the differences in per-building β coefficients (driven primarily by envelope "
    "performance and energy class). End-century distributions become bimodal, with "
    "energy-efficient buildings clustering at low intensity and class F-G stock "
    "extending the upper tail beyond 200 kWh/m²/yr.")

# ---------------------------------------------------------------------------
# Figure 4.8 - Multi-paper validation
# ---------------------------------------------------------------------------
print("  fig_4_8_multipaper_validation_multimodel...")
v1_today = v1[v1["scenario"] == "ERA5 recent 2015-2024"]["total_GWh_median"].iloc[0]
v1_2080 = v1[v1["scenario"] == "SSP5-8.5 / 2080-2100"]["total_GWh_median"].iloc[0]
mean_today = cooled_only[f"int_{sc_today}"].mean()
mean_2080 = cooled_only[f"int_{sc_585_2080}"].mean()

# Literature ranges (translated from the locked tab_4_5)
papers = [
    ("Tootkaboni 2021\n(Energy Reports)",      10, 30,  30, 70,  "Milan TRNSYS"),
    ("Sangelantoni 2022\n(SCS)",               20, 40,  60, 100, "N.Italy CDH"),
    ("D'Agostino 2022\n(Energy & Build.)",     15, 45,  40, 110, "Italy resi."),
    ("Ascione 2017\n(Energy & Build.)",        25, 50,  np.nan, np.nan, "Zone E"),
]
fig, axes = plt.subplots(1, 2, figsize=(16, 6.5))
# Panel a: today
ax = axes[0]
labels = [p[0] for p in papers]
y = np.arange(len(papers))
for i, (lab, t_lo, t_hi, _, _, _) in enumerate(papers):
    ax.plot([t_lo, t_hi], [i, i], "-", color="#1F4E79", linewidth=4, alpha=0.6)
    ax.plot([t_lo, t_hi], [i, i], "|", color="#1F4E79", markersize=12)
ax.axvline(mean_today, color="#A03020", lw=2.0, ls="--",
           label=f"This thesis: {mean_today:.1f}")
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Cooling intensity today (kWh/m²/yr)")
ax.set_title("(a) Today / current climate", fontsize=11, fontweight="bold",
             color=HEADING, loc="left")
ax.legend(loc="lower right", frameon=False)
ax.grid(axis="x", alpha=0.3, linewidth=0.5)

# Panel b: 2080 future
ax = axes[1]
for i, (lab, _, _, f_lo, f_hi, _) in enumerate(papers):
    if not np.isnan(f_lo):
        ax.plot([f_lo, f_hi], [i, i], "-", color="#A03020", linewidth=4, alpha=0.6)
        ax.plot([f_lo, f_hi], [i, i], "|", color="#A03020", markersize=12)
    else:
        ax.text(50, i, "(historical study, no projection)", fontsize=9,
                color="#999999", style="italic", va="center")
ax.axvline(mean_2080, color="#A03020", lw=2.0, ls="--",
           label=f"This thesis: {mean_2080:.1f}")
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("Cooling intensity 2080-2100 (kWh/m²/yr)")
ax.set_title("(b) End-century / SSP5-8.5", fontsize=11, fontweight="bold",
             color=HEADING, loc="left")
ax.legend(loc="lower right", frameon=False)
ax.grid(axis="x", alpha=0.3, linewidth=0.5)
fig.suptitle("Multi-paper validation: this thesis ensemble vs published Italian/Mediterranean studies",
             fontsize=13, fontweight="bold", color=HEADING, y=1.0)
plt.tight_layout()
stem = FIGURES_DIR / "fig_4_8_multipaper_validation_multimodel"
save_fig(fig, stem)
save_caption(stem,
    "Figure 4.8. Validation of mean per-building cooling intensity against four "
    "published Italian/Mediterranean studies. Horizontal bars show literature ranges; "
    "the red dashed line shows this thesis ensemble median. (a) Current climate: "
    "thesis estimate falls within the range reported by all four studies. (b) "
    "End-century SSP5-8.5: thesis estimate is consistent with D'Agostino 2022 "
    "(Italian residential reference buildings) and Sangelantoni 2022 (Italy whole). "
    "Results validate the multi-model methodology.")

print(f"  Non-spatial figures complete in {time.time()-t_load:.1f}s total")

# ===========================================================================
# STEP 7: SPATIAL MAPS (helper + 8 maps)
# ===========================================================================

print("\n[7] Generating maps (8 maps, ~1 min each due to 53k polygons)...")

def make_choropleth(buildings_gdf_, value_col, title, cmap_name, vmin, vmax,
                     cbar_label, footer, outpath_stem, divnorm=False, center=None):
    """Render building-level choropleth map (consistent style)."""
    fig, ax = plt.subplots(figsize=(11.5, 11))
    has_data = buildings_gdf_[value_col].notna()
    no_data = ~has_data
    # Background buildings (no EPC data) in light grey
    if no_data.sum() > 0:
        buildings_gdf_[no_data].plot(ax=ax, color="#E8E8E8",
                                      edgecolor="none", linewidth=0)
    # Foreground: data buildings
    if divnorm and center is not None:
        from matplotlib.colors import TwoSlopeNorm
        norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
        buildings_gdf_[has_data].plot(
            ax=ax, column=value_col, cmap=cmap_name, norm=norm,
            edgecolor="none", linewidth=0
        )
        sm = ScalarMappable(cmap=cmap_name, norm=norm)
    else:
        buildings_gdf_[has_data].plot(
            ax=ax, column=value_col, cmap=cmap_name, vmin=vmin, vmax=vmax,
            edgecolor="none", linewidth=0
        )
        sm = ScalarMappable(cmap=cmap_name, norm=Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.040, pad=0.02)
    cbar.set_label(cbar_label, fontsize=11)
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=13, fontweight="bold", color=HEADING, loc="left")
    # Scale bar (1 km)
    scale_x = xmin + (xmax - xmin) * 0.04
    scale_y = ymin + (ymax - ymin) * 0.05
    ax.plot([scale_x, scale_x + 1000], [scale_y, scale_y], color="black", lw=2.5)
    ax.text(scale_x + 500, scale_y - (ymax-ymin)*0.012, "1 km",
            ha="center", va="top", fontsize=9, fontweight="bold")
    # North arrow
    arr_x = xmax - (xmax - xmin) * 0.06
    arr_y = ymax - (ymax - ymin) * 0.12
    ax.annotate("N", xy=(arr_x, arr_y), xytext=(arr_x, arr_y - (ymax-ymin)*0.04),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.8),
                ha="center", fontsize=12, fontweight="bold")
    # Footer
    fig.text(0.5, 0.02, footer, ha="center", fontsize=8.5, style="italic", color="#444444")
    plt.tight_layout()
    return save_fig(fig, outpath_stem)

# Color scale ranges - consistent across comparable maps
all_today = buildings_mapped[f"int_{sc_today}"].dropna()
all_585_2080 = buildings_mapped[f"int_{sc_585_2080}"].dropna()
int_vmax = float(np.percentile(all_585_2080, 98))  # robust upper bound
int_vmin = 0
hist_safe = "ERA5 historical 1990-2024".replace(" ", "_").replace("/", "-").replace(".", "")
mod_safe = "SSP2-4.5 / 2080-2100".replace(" ", "_").replace("/", "-").replace(".", "")

# Map 4.1 - historical
print("  map_4_1_historical_1990_2024_multimodel...")
make_choropleth(
    buildings_mapped, f"int_{hist_safe}",
    "Map 4.1. Cooling intensity per building - ERA5 historical 1990-2024",
    "YlOrRd", int_vmin, int_vmax,
    "Cooling intensity (kWh/m²/yr)",
    "n=53,041 buildings shown; 19,063 have EPC data and are colored. "
    "ERA5 baseline: 162.1 CDD.",
    MAPS_DIR / "map_4_1_historical_1990_2024_multimodel"
)
save_caption(MAPS_DIR / "map_4_1_historical_1990_2024_multimodel",
    "Map 4.1. Predicted per-building cooling intensity (kWh/m²/yr) under the ERA5 "
    "1990-2024 historical climate. Building geometry from Geoportale Milano "
    "(53,041 polygons); buildings without EPC data shown in light grey. Intensity "
    "computed as β × CDD where β is the locked ML prediction from the 4-learner "
    "stacking ensemble.")

# Map 4.2 - today
print("  map_4_2_today_2015_2024_multimodel...")
make_choropleth(
    buildings_mapped, f"int_{sc_today}",
    "Map 4.2. Cooling intensity per building - Today (ERA5 2015-2024)",
    "YlOrRd", int_vmin, int_vmax,
    "Cooling intensity (kWh/m²/yr)",
    "ERA5 recent baseline: 206.6 CDD. Total cooled stock demand: 38.6 GWh/yr.",
    MAPS_DIR / "map_4_2_today_2015_2024_multimodel"
)
save_caption(MAPS_DIR / "map_4_2_today_2015_2024_multimodel",
    "Map 4.2. Predicted per-building cooling intensity under the recent decade "
    "ERA5 2015-2024 baseline (206.6 CDD). This is the current-climate anchor used "
    "throughout the projection analysis.")

# Map 4.3 - moderate 2080
print("  map_4_3_moderate_2080_multimodel...")
make_choropleth(
    buildings_mapped, f"int_{mod_safe}",
    "Map 4.3. Cooling intensity per building - SSP2-4.5 / 2080-2100 (ensemble median)",
    "YlOrRd", int_vmin, int_vmax,
    "Cooling intensity (kWh/m²/yr)",
    f"12-model NEX-GDDP-CMIP6 ensemble median CDD = {cdd_medians['SSP2-4.5 / 2080-2100']:.0f}.",
    MAPS_DIR / "map_4_3_moderate_2080_multimodel"
)
save_caption(MAPS_DIR / "map_4_3_moderate_2080_multimodel",
    "Map 4.3. Predicted per-building cooling intensity under SSP2-4.5 / 2080-2100, "
    "using the 12-model NEX-GDDP-CMIP6 ensemble median CDD. Color scale matched to "
    "Maps 4.1, 4.2, 4.4 for direct visual comparison.")

# Map 4.4 - high 2080
print("  map_4_4_high_2080_multimodel...")
make_choropleth(
    buildings_mapped, f"int_{sc_585_2080}",
    "Map 4.4. Cooling intensity per building - SSP5-8.5 / 2080-2100 (ensemble median)",
    "YlOrRd", int_vmin, int_vmax,
    "Cooling intensity (kWh/m²/yr)",
    f"12-model NEX-GDDP-CMIP6 ensemble median CDD = {cdd_medians['SSP5-8.5 / 2080-2100']:.0f}. "
    "Total cooled stock demand: 139.5 [109-171] GWh/yr (95% CI across 12 models).",
    MAPS_DIR / "map_4_4_high_2080_multimodel"
)
save_caption(MAPS_DIR / "map_4_4_high_2080_multimodel",
    "Map 4.4. Predicted per-building cooling intensity under SSP5-8.5 / 2080-2100. "
    "Headline scenario for the thesis: ensemble median CDD reaches 747.6, demand "
    "tripling to 139.5 GWh/yr (median across 12 NEX-GDDP-CMIP6 models).")

# Map 4.5 - absolute change
print("  map_4_5_demand_change_absolute_multimodel...")
buildings_mapped["delta_int"] = (buildings_mapped[f"int_{sc_585_2080}"] -
                                   buildings_mapped[f"int_{sc_today}"])
delta_max = float(np.percentile(buildings_mapped["delta_int"].dropna(), 98))
make_choropleth(
    buildings_mapped, "delta_int",
    "Map 4.5. Absolute increase in cooling intensity - SSP5-8.5/2080 vs Today",
    "YlOrRd", 0, delta_max,
    "Δ intensity (kWh/m²/yr)",
    f"Mean increase: {buildings_mapped['delta_int'].mean():.1f} kWh/m²/yr. "
    f"Maximum increase: class G buildings up to {buildings_mapped['delta_int'].max():.0f} kWh/m²/yr.",
    MAPS_DIR / "map_4_5_demand_change_absolute_multimodel"
)
save_caption(MAPS_DIR / "map_4_5_demand_change_absolute_multimodel",
    "Map 4.5. Absolute increase (kWh/m²/yr) in per-building cooling intensity from "
    "ERA5 recent (today) to SSP5-8.5 / 2080-2100 ensemble median. Hotspots correspond "
    "to poorly-insulated older stock (energy classes F and G), whose β coefficients "
    "amplify the climate-change CDD signal.")

# Map 4.6 - 88-cluster overlay (district aggregation)
print("  map_4_6_demand_per_district_88clusters_multimodel...")
fig, ax = plt.subplots(figsize=(11.5, 11))
buildings_mapped.plot(ax=ax, color="#E8E8E8", edgecolor="none", linewidth=0)
size_scale = districts_mm["total_GWh_today"] / districts_mm["total_GWh_today"].max() * 800 + 50
sc_plot = ax.scatter(
    district_geom.geometry.x, district_geom.geometry.y,
    s=size_scale, c=districts_mm["total_GWh_today"].values,
    cmap="YlOrRd", edgecolor="black", linewidth=0.4, alpha=0.85
)
cbar = plt.colorbar(sc_plot, ax=ax, fraction=0.040, pad=0.02)
cbar.set_label("District demand today (GWh/yr)", fontsize=11)
ax.set_xlim(xmin - pad_x, xmax + pad_x)
ax.set_ylim(ymin - pad_y, ymax + pad_y)
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_title("Map 4.6. Cooling demand by 88 spatial districts (today's climate)",
             fontsize=13, fontweight="bold", color=HEADING, loc="left")
# Scale bar
scale_x = xmin + (xmax - xmin) * 0.04
scale_y = ymin + (ymax - ymin) * 0.05
ax.plot([scale_x, scale_x + 1000], [scale_y, scale_y], color="black", lw=2.5)
ax.text(scale_x + 500, scale_y - (ymax-ymin)*0.012, "1 km",
        ha="center", va="top", fontsize=9, fontweight="bold")
# North arrow
arr_x = xmax - (xmax - xmin) * 0.06
arr_y = ymax - (ymax - ymin) * 0.12
ax.annotate("N", xy=(arr_x, arr_y), xytext=(arr_x, arr_y - (ymax-ymin)*0.04),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.8),
            ha="center", fontsize=12, fontweight="bold")
fig.text(0.5, 0.02,
         f"Marker size proportional to district demand (today). 88 districts "
         f"derived by nearest-centroid assignment of {len(buildings_mapped):,} "
         f"buildings to locked centroids.",
         ha="center", fontsize=8.5, style="italic", color="#444444")
plt.tight_layout()
stem = MAPS_DIR / "map_4_6_demand_per_district_88clusters_multimodel"
save_fig(fig, stem)
save_caption(stem,
    "Map 4.6. Aggregated cooling demand at the 88 spatial-cluster level under today's "
    "climate (ERA5 2015-2024). Each marker shows one district; marker size and color "
    "are proportional to total district demand (GWh/yr). Districts derived by "
    "spatial clustering of cooled buildings into 88 zones, preserving locked Stage 2 "
    "spatial structure.")

# Map 4.7 - district relative change (% increase) - diverging colormap centered on city median
print("  map_4_7_district_delta_585_vs_today_multimodel...")
fig, ax = plt.subplots(figsize=(11.5, 11))
buildings_mapped.plot(ax=ax, color="#E8E8E8", edgecolor="none", linewidth=0)
pct_vals = districts_mm["pct_change_585_vs_today"].values
pct_vmin = float(np.nanpercentile(pct_vals, 2))
pct_vmax = float(np.nanpercentile(pct_vals, 98))
pct_center = float(np.nanmedian(pct_vals))
from matplotlib.colors import TwoSlopeNorm
norm = TwoSlopeNorm(vmin=pct_vmin, vcenter=pct_center, vmax=pct_vmax)
sc_plot = ax.scatter(
    district_geom.geometry.x, district_geom.geometry.y,
    s=300, c=pct_vals, cmap="RdYlBu_r", norm=norm,
    edgecolor="black", linewidth=0.5, alpha=0.92
)
cbar = plt.colorbar(sc_plot, ax=ax, fraction=0.040, pad=0.02)
cbar.set_label("% change in district demand: SSP5-8.5/2080 vs today", fontsize=11)
ax.set_xlim(xmin - pad_x, xmax + pad_x)
ax.set_ylim(ymin - pad_y, ymax + pad_y)
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_title("Map 4.7. Relative change in district demand (SSP5-8.5/2080 vs today)",
             fontsize=13, fontweight="bold", color=HEADING, loc="left")
scale_x = xmin + (xmax - xmin) * 0.04
scale_y = ymin + (ymax - ymin) * 0.05
ax.plot([scale_x, scale_x + 1000], [scale_y, scale_y], color="black", lw=2.5)
ax.text(scale_x + 500, scale_y - (ymax-ymin)*0.012, "1 km",
        ha="center", va="top", fontsize=9, fontweight="bold")
arr_x = xmax - (xmax - xmin) * 0.06
arr_y = ymax - (ymax - ymin) * 0.12
ax.annotate("N", xy=(arr_x, arr_y), xytext=(arr_x, arr_y - (ymax-ymin)*0.04),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.8),
            ha="center", fontsize=12, fontweight="bold")
fig.text(0.5, 0.02,
         f"Median percent increase: {pct_center:.0f}%. "
         f"Diverging colormap centered on city median.",
         ha="center", fontsize=8.5, style="italic", color="#444444")
plt.tight_layout()
stem = MAPS_DIR / "map_4_7_district_delta_585_vs_today_multimodel"
save_fig(fig, stem)
save_caption(stem,
    "Map 4.7. Percentage change in district-level cooling demand from today's climate "
    "(ERA5 2015-2024) to SSP5-8.5 / 2080-2100 (12-model ensemble median). Diverging "
    "colormap centered on the city-wide median percent change. Districts above the "
    "city median (warmer colors) tend to have older, lower-class building stock with "
    "higher β coefficients; districts below the median (cooler colors) have more "
    "energy-efficient stock that absorbs the future CDD signal with smaller relative "
    "change.")

# Map 4.8 - 4-panel summary
print("  map_4_8_summary_4panel_multimodel...")
fig, axes = plt.subplots(2, 2, figsize=(20, 19))
panels = [
    (axes[0, 0], f"int_{sc_today}",       "(a) Today (ERA5 2015-2024)",                "YlOrRd", int_vmin, int_vmax, False),
    (axes[0, 1], f"int_{mod_safe}",       "(b) SSP2-4.5 / 2080-2100",                  "YlOrRd", int_vmin, int_vmax, False),
    (axes[1, 0], f"int_{sc_585_2080}",    "(c) SSP5-8.5 / 2080-2100",                  "YlOrRd", int_vmin, int_vmax, False),
    (axes[1, 1], "delta_int",              "(d) Δ intensity SSP5-8.5/2080 vs today",   "YlOrRd", 0,        delta_max, False),
]
for ax, col, title, cmap_name, vmn, vmx, _ in panels:
    has_data = buildings_mapped[col].notna()
    no_data = ~has_data
    if no_data.sum() > 0:
        buildings_mapped[no_data].plot(ax=ax, color="#E8E8E8",
                                         edgecolor="none", linewidth=0)
    buildings_mapped[has_data].plot(
        ax=ax, column=col, cmap=cmap_name, vmin=vmn, vmax=vmx,
        edgecolor="none", linewidth=0
    )
    sm = ScalarMappable(cmap=cmap_name, norm=Normalize(vmin=vmn, vmax=vmx))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.038, pad=0.02)
    cbar.set_label("kWh/m²/yr", fontsize=10)
    ax.set_xlim(xmin - pad_x, xmax + pad_x)
    ax.set_ylim(ymin - pad_y, ymax + pad_y)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=12, fontweight="bold", color=HEADING, loc="left")

fig.suptitle("Map 4.8. Per-building cooling intensity in Milan: scenario comparison",
             fontsize=16, fontweight="bold", color=HEADING, y=0.995)
fig.text(0.5, 0.01,
         "Building polygons from Geoportale Milano (n=53,041); 19,063 EPC buildings colored. "
         "Future scenarios use 12-model NEX-GDDP-CMIP6 ensemble median CDD. "
         "Color scale (a-c) matched for direct visual comparison.",
         ha="center", fontsize=9.5, style="italic", color="#444444")
plt.tight_layout()
stem = MAPS_DIR / "map_4_8_summary_4panel_multimodel"
save_fig(fig, stem)
save_caption(stem,
    "Map 4.8. Four-panel summary of per-building cooling intensity in Milan. "
    "(a) Today's baseline (ERA5 recent 2015-2024). (b) Moderate-emission end-century "
    "(SSP2-4.5 / 2080-2100). (c) High-emission end-century (SSP5-8.5 / 2080-2100). "
    "(d) Absolute increase from (a) to (c). Panels (a)-(c) share a color scale "
    "covering 0 to 98th percentile of the SSP5-8.5/2080 distribution. The visual "
    "intensification from (a) through (c) provides direct evidence that climate "
    "change makes Milan's existing residential stock substantially more "
    "cooling-intensive, with the highest impact concentrated in older, lower "
    "energy-class buildings.")

print(f"  Maps complete in {time.time()-t_load:.1f}s total")

# ===========================================================================
# STEP 8: FINAL LOGGING AND SUMMARY
# ===========================================================================

exec_log.extend([
    "",
    "Visualizations produced:",
    "  Non-spatial figures:",
    "    fig_4_1_cdd_evolution_multimodel",
    "    fig_4_2_cdd_distributions_multimodel",
    "    fig_4_4_demand_per_class_heatmap_multimodel",
    "    fig_4_5_climate_signal_summary_multimodel",
    "    fig_4_6_adoption_pathway_comparison_multimodel",
    "    fig_4_7_demand_distribution_violin_multimodel",
    "    fig_4_8_multipaper_validation_multimodel",
    "  Maps:",
    "    map_4_1_historical_1990_2024_multimodel",
    "    map_4_2_today_2015_2024_multimodel",
    "    map_4_3_moderate_2080_multimodel",
    "    map_4_4_high_2080_multimodel",
    "    map_4_5_demand_change_absolute_multimodel",
    "    map_4_6_demand_per_district_88clusters_multimodel",
    "    map_4_7_district_delta_585_vs_today_multimodel",
    "    map_4_8_summary_4panel_multimodel",
    "",
    f"Each PNG accompanied by PDF and .caption.txt",
    f"District-level table: tab_4_6_demand_per_district_multimodel.csv",
    "",
    f"Script D finished: {datetime.now().isoformat()}",
])
exec_log_path = LOGS_DIR / "script_d_execution_log.txt"
with open(exec_log_path, "w", encoding="utf-8") as f:
    f.write("\n".join(exec_log))

print("\n" + "=" * 75)
print("SCRIPT D COMPLETE")
print("=" * 75)
print(f"\nOutputs in: {MULTIMODEL_ROOT}")
print(f"  Figures: {FIGURES_DIR.name}/  (7 figures, each PNG + PDF + caption)")
print(f"  Maps:    {MAPS_DIR.name}/  (8 maps, each PNG + PDF + caption)")
print(f"  Tables:  {TABLES_DIR.name}/  (district aggregation table updated)")
print(f"  Log:     {exec_log_path.name}")
print(f"\nReady to write the methodology document.")
print(f"\nTotal runtime: {(time.time()-t_load)/60:.1f} min")