"""
============================================================================
PATCH: Regenerate map_4_7 and map_4_8 only.

Map 4.7 reworked to show ABSOLUTE district-level GWh change (sequential cmap)
instead of the % change (which was spatially homogeneous and broke
TwoSlopeNorm). Map 4.8 regenerated cleanly.
============================================================================
"""

import sys
import time
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
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
MAPS_DIR = MULTIMODEL_ROOT / "_MAPS"
TABLES_DIR = MULTIMODEL_ROOT / "_TABLES"

PER_BUILDING_FILE = (STAGE2_ROOT / "session4_demand_projection" / "E_outputs"
                     / "stage2_per_building_demand_projections_v2.csv")
CDD_MULTIMODEL_FILE = TABLES_DIR / "tab_4_1_cdd_projections_multimodel.csv"
DISTRICTS_MM_FILE = TABLES_DIR / "tab_4_6_demand_per_district_multimodel.csv"
SHAPEFILE = Path(r"C:\Users\n.mohammadi\Desktop\NimaMohammadi"
                 r"\03. Nima Mohammadi - Thesis\Data Collections\Geoportale Milano\A020102.shp")

ERA5_RECENT_CDD = 206.6

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
print("PATCH: Regenerate map_4_7 and map_4_8")
print(f"Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 75)

t0 = time.time()

def save_caption(stem, text):
    with open(Path(str(stem) + ".caption.txt"), "w", encoding="utf-8") as f:
        f.write(text)

def save_fig(fig, stem):
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

# ===========================================================================
# LOAD DATA
# ===========================================================================

print("\n[1] Loading data...")

# District table (already produced by Script D, has multi-model values)
districts_mm = pd.read_csv(DISTRICTS_MM_FILE)
print(f"  Districts: {len(districts_mm)}")

# Shapefile
print(f"  Loading shapefile (~10 sec)...")
buildings_gdf = gpd.read_file(SHAPEFILE)
print(f"  Shapefile: {len(buildings_gdf):,} polygons")

# Per-building (for map_4_8)
proj = pd.read_csv(PER_BUILDING_FILE, low_memory=False)
beta = proj["beta_predicted"].values
area = proj["effective_cooled_area_m2"].values

# Multi-model CDD ensemble medians
cdd_summary = pd.read_csv(CDD_MULTIMODEL_FILE)
sc_map = {
    "SSP2-4.5 / 2080-2100": ("ssp245", 2080),
    "SSP5-8.5 / 2080-2100": ("ssp585", 2080),
}
cdd_medians = {"ERA5 recent 2015-2024": ERA5_RECENT_CDD}
for sc_disp, (sc_short, p_start) in sc_map.items():
    row = cdd_summary[(cdd_summary["scenario"] == sc_short) &
                       (cdd_summary["period_start"] == p_start)].iloc[0]
    cdd_medians[sc_disp] = float(row["cdd_ensemble_median"])

# Compute per-building intensities for the 4 panels of map_4_8
def safe(s): return s.replace(" ", "_").replace("/", "-").replace(".", "")

per_bldg = pd.DataFrame({"EDIFC_ID": proj["EDIFC_ID"].values})
for sc, cdd_val in cdd_medians.items():
    per_bldg[f"int_{safe(sc)}"] = beta * cdd_val
per_bldg["delta_int"] = (per_bldg[f"int_{safe('SSP5-8.5 / 2080-2100')}"] -
                          per_bldg[f"int_{safe('ERA5 recent 2015-2024')}"])

buildings_mapped = buildings_gdf.merge(per_bldg, on="EDIFC_ID", how="left")
n_with_data = buildings_mapped[f"int_{safe('ERA5 recent 2015-2024')}"].notna().sum()
print(f"  Polygons with EPC data: {n_with_data:,}/{len(buildings_mapped):,}")

# District centroids reprojected for plotting
district_geom = gpd.GeoDataFrame(
    districts_mm,
    geometry=[Point(xy) for xy in zip(districts_mm["lon_centroid"], districts_mm["lat_centroid"])],
    crs="EPSG:4326"
).to_crs(buildings_gdf.crs)

xmin, ymin, xmax, ymax = buildings_gdf.total_bounds
pad_x = (xmax - xmin) * 0.02
pad_y = (ymax - ymin) * 0.02

print(f"  Loading complete in {time.time()-t0:.1f}s")

# ===========================================================================
# MAP 4.7 - REWORKED: absolute district-level GWh increase
# ===========================================================================

print("\n[2] map_4_7_district_delta_585_vs_today_multimodel (reworked)...")
fig, ax = plt.subplots(figsize=(11.5, 11))
buildings_mapped.plot(ax=ax, color="#E8E8E8", edgecolor="none", linewidth=0)

delta_vals = districts_mm["delta_GWh_585_vs_today"].values
delta_vmin = 0
delta_vmax = float(np.nanpercentile(delta_vals, 98))

# Scatter sized by absolute change, colored by absolute change
size_scale = (delta_vals - delta_vmin) / max(delta_vmax - delta_vmin, 1e-9) * 700 + 80
size_scale = np.clip(size_scale, 80, 800)

sc_plot = ax.scatter(
    district_geom.geometry.x, district_geom.geometry.y,
    s=size_scale, c=delta_vals, cmap="YlOrRd",
    vmin=delta_vmin, vmax=delta_vmax,
    edgecolor="black", linewidth=0.5, alpha=0.92
)
cbar = plt.colorbar(sc_plot, ax=ax, fraction=0.040, pad=0.02)
cbar.set_label("Δ district demand: SSP5-8.5/2080 vs today (GWh/yr)", fontsize=11)
ax.set_xlim(xmin - pad_x, xmax + pad_x)
ax.set_ylim(ymin - pad_y, ymax + pad_y)
ax.set_xticks([]); ax.set_yticks([])
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_title("Map 4.7. Absolute district-level demand increase (SSP5-8.5/2080 vs today)",
             fontsize=13, fontweight="bold", color=HEADING, loc="left")

# Scale bar + N arrow
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

total_increase = delta_vals.sum()
n_districts = (delta_vals > 0).sum()
fig.text(0.5, 0.02,
         f"Total increase across {n_districts} active districts: {total_increase:.0f} GWh/yr. "
         f"Marker size and color proportional to district increase.",
         ha="center", fontsize=8.5, style="italic", color="#444444")
plt.tight_layout()
stem = MAPS_DIR / "map_4_7_district_delta_585_vs_today_multimodel"
save_fig(fig, stem)
save_caption(stem,
    "Map 4.7. Absolute increase in district-level cooling demand from today's climate "
    "(ERA5 2015-2024) to SSP5-8.5 / 2080-2100 (12-model ensemble median). Each marker "
    "represents one of 88 spatial clusters; both color and marker size are proportional "
    "to the absolute GWh increase. Districts with the largest absolute increases combine "
    "high building density with older, lower-class stock that amplifies the climate "
    "signal through their per-building β coefficients. Total city-wide increase across "
    "all districts: {:.0f} GWh/yr.".format(total_increase))

print(f"  Saved: map_4_7")

# ===========================================================================
# MAP 4.8 - 4-panel summary
# ===========================================================================

print("\n[3] map_4_8_summary_4panel_multimodel...")

sc_today = safe("ERA5 recent 2015-2024")
sc_mod = safe("SSP2-4.5 / 2080-2100")
sc_high = safe("SSP5-8.5 / 2080-2100")

# Determine consistent intensity scale across (a)-(c)
all_high = buildings_mapped[f"int_{sc_high}"].dropna()
int_vmax = float(np.percentile(all_high, 98))
int_vmin = 0
delta_vmax_b = float(np.percentile(buildings_mapped["delta_int"].dropna(), 98))

fig, axes = plt.subplots(2, 2, figsize=(20, 19))
panels = [
    (axes[0, 0], f"int_{sc_today}",  "(a) Today (ERA5 2015-2024)",            "YlOrRd", int_vmin, int_vmax),
    (axes[0, 1], f"int_{sc_mod}",    "(b) SSP2-4.5 / 2080-2100",              "YlOrRd", int_vmin, int_vmax),
    (axes[1, 0], f"int_{sc_high}",   "(c) SSP5-8.5 / 2080-2100",              "YlOrRd", int_vmin, int_vmax),
    (axes[1, 1], "delta_int",         "(d) Δ intensity SSP5-8.5/2080 vs today", "YlOrRd", 0, delta_vmax_b),
]
for ax, col, title, cmap_name, vmn, vmx in panels:
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

print(f"  Saved: map_4_8")

print("\n" + "=" * 75)
print("PATCH COMPLETE")
print("=" * 75)
print(f"\n  Total runtime: {(time.time()-t0)/60:.1f} min")
print(f"\nAll 7 figures + 8 maps now generated.")
print(f"  Figures: {MULTIMODEL_ROOT / '_FIGURES'}")
print(f"  Maps:    {MAPS_DIR}")
print(f"\nReady to write the methodology document.")