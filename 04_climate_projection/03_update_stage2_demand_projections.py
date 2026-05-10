"""
============================================================================
SCRIPT B: Update Session 4 Demand Projections with Multi-Model CMIP6
----------------------------------------------------------------------------
This single script does everything for the Session 4 multi-model revision:

  1. Loads the locked per-building beta predictions (19,063 buildings)
  2. Loads the multi-model CDD table from Script A
  3. Loops over 12 models x 4 future scenarios + 2 ERA5 anchors
  4. Computes per-building demand for each (model, scenario)
  5. Aggregates to V1 (cooled), V2 (full), V3 (adoption pathways)
  6. Computes ensemble statistics (median, p5, p95) across 12 models
  7. Generates updated Figure 4.3 with multi-model error bars
  8. Saves all outputs to _MULTIMODEL/_TABLES/ and _MULTIMODEL/_FIGURES/

Methodological notes:
  - beta_predicted is LOCKED from Stage 2 trained models (no re-prediction)
  - Demand is exactly linear in CDD: demand = beta * cdd * area
  - ERA5 baseline kept as deterministic observation anchor (162.1, 206.6 CDD)
  - 12-model NEX-GDDP-CMIP6 ensemble used for 4 future scenarios
  - Adoption pathways for V3: 72%, 85%, 100% (consistent with locked Stage 2)

Runtime: 1-2 minutes
Idempotent: re-running overwrites cleanly
============================================================================
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import sys

# ===========================================================================
# CONFIGURATION
# ===========================================================================

# Try both old and new project root paths (user has been switching)
CANDIDATE_PROJECT_ROOTS = [
    Path(r"C:\Users\n.mohammadi\Desktop\NimaMohammadi\03. Nima Mohammadi - Thesis\ML Dataset"),
    Path(r"C:\Users\n.mohammadi\Desktop\NimaMohammadi\02.Nima Mohammadi - Thesis\ML Dataset"),
]

# Detect which path actually exists
PROJECT_ROOT = None
for cand in CANDIDATE_PROJECT_ROOTS:
    if cand.exists():
        PROJECT_ROOT = cand
        break
if PROJECT_ROOT is None:
    print("ERROR: Cannot find ML Dataset folder at any candidate path.")
    sys.exit(1)

print(f"Using project root: {PROJECT_ROOT}")

STAGE2_ROOT = PROJECT_ROOT / "07_results" / "stage2"
MULTIMODEL_ROOT = STAGE2_ROOT / "_MULTIMODEL"

# Inputs
PER_BUILDING_FILE = (STAGE2_ROOT / "session4_demand_projection" / "E_outputs"
                     / "stage2_per_building_demand_projections_v2.csv")
CDD_MULTIMODEL_FILE = MULTIMODEL_ROOT / "_TABLES" / "tab_4_1_cdd_projections_multimodel.csv"

# Outputs (inside _MULTIMODEL)
TABLES_DIR = MULTIMODEL_ROOT / "_TABLES"
FIGURES_DIR = MULTIMODEL_ROOT / "_FIGURES"
LOGS_DIR = MULTIMODEL_ROOT / "_LOGS"
for d in [TABLES_DIR, FIGURES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Methodology constants (LOCKED from existing Stage 2)
ERA5_HISTORICAL_CDD = 162.1   # ERA5 1990-2024 mean
ERA5_RECENT_CDD = 206.6        # ERA5 2015-2024 mean

# Models (must match Script A)
MODELS = [
    "MPI-ESM1-2-HR", "CMCC-ESM2", "EC-Earth3", "CNRM-ESM2-1",
    "IPSL-CM6A-LR", "CESM2", "HadGEM3-GC31-LL", "CanESM5",
    "NorESM2-MM", "GFDL-ESM4", "MIROC6", "ACCESS-ESM1-5",
]

# Future scenarios (will use 12-model ensemble)
FUTURE_SCENARIOS = [
    ("ssp245", "2030-2050", 2030, 2050, "SSP2-4.5 / 2030-2050"),
    ("ssp245", "2080-2100", 2080, 2100, "SSP2-4.5 / 2080-2100"),
    ("ssp585", "2030-2050", 2030, 2050, "SSP5-8.5 / 2030-2050"),
    ("ssp585", "2080-2100", 2080, 2100, "SSP5-8.5 / 2080-2100"),
]

# ERA5 reference scenarios (deterministic, single value each)
ERA5_SCENARIOS = [
    ("ERA5 historical 1990-2024", ERA5_HISTORICAL_CDD),
    ("ERA5 recent 2015-2024", ERA5_RECENT_CDD),
]

# V3 adoption pathways
ADOPTION = {"V3_Conservative_72pct": 0.72, "V3_Moderate_85pct": 0.85, "V3_High_100pct": 1.00}

# Plot setup
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

# ===========================================================================
# HEADER
# ===========================================================================

print("=" * 75)
print("SCRIPT B: Multi-Model Demand Projections (Session 4 Revision)")
print(f"Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 75)

exec_log = [
    "Script B execution log",
    f"Started: {datetime.now().isoformat()}",
    f"Project root: {PROJECT_ROOT}",
    "=" * 75,
    "",
]

# ===========================================================================
# STEP 1: LOAD INPUTS
# ===========================================================================

print("\n[1] Loading inputs...")

if not PER_BUILDING_FILE.exists():
    print(f"  ERROR: per-building file not found at {PER_BUILDING_FILE}")
    sys.exit(1)

per_building = pd.read_csv(PER_BUILDING_FILE, low_memory=False)
print(f"  Loaded per-building file: {per_building.shape[0]:,} buildings, {per_building.shape[1]} columns")

# Verify required columns
required = ["EDIFC_ID", "beta_predicted", "effective_cooled_area_m2",
            "CLIMATIZZAZIONE_ESTIVA", "CLASSE_ENERGETICA"]
missing = [c for c in required if c not in per_building.columns]
if missing:
    print(f"  ERROR: missing columns: {missing}")
    sys.exit(1)

# Extract working arrays
beta = per_building["beta_predicted"].values
area = per_building["effective_cooled_area_m2"].values
is_cooled = (per_building["CLIMATIZZAZIONE_ESTIVA"] == True).values
energy_class = per_building["CLASSE_ENERGETICA"].values
n_total = len(per_building)
n_cooled = is_cooled.sum()
print(f"  Total buildings: {n_total:,}")
print(f"  Cooled (V1 mask): {n_cooled:,}")

if not CDD_MULTIMODEL_FILE.exists():
    print(f"  ERROR: multi-model CDD table not found at {CDD_MULTIMODEL_FILE}")
    print(f"  Run Script A (validate_and_build_cdd.py) first.")
    sys.exit(1)

cdd_table = pd.read_csv(CDD_MULTIMODEL_FILE)
print(f"  Loaded multi-model CDD table: {len(cdd_table)} rows")
exec_log.append(f"Inputs loaded successfully")

# ===========================================================================
# STEP 2: HELPER FUNCTIONS
# ===========================================================================

def compute_variants_for_cdd(cdd_value):
    """Given a single CDD value, compute V1/V2/V3 totals and average intensity.

    Returns dict with total_GWh and avg_kWh_per_bldg for each variant.
    """
    intensity = beta * cdd_value         # kWh/m^2/yr per building
    total = intensity * area              # kWh/yr per building

    # V1: cooled stock
    v1_total_GWh = total[is_cooled].sum() / 1e6
    v1_avg = total[is_cooled].mean()

    # V2: full stock (all 19,063 buildings, treated as cooled)
    v2_total_GWh = total.sum() / 1e6
    v2_avg = total.mean()

    # V3: adoption pathways (V2 scaled by adoption rate)
    v3_results = {}
    for name, rate in ADOPTION.items():
        v3_results[name] = {
            "total_GWh": v2_total_GWh * rate,
            "avg_kWh_per_bldg": v2_avg,  # average per building unchanged
            "n_active": int(n_total * rate),
            "adoption": rate,
        }

    return {
        "V1_cooled_n13787": {"total_GWh": v1_total_GWh, "avg_kWh_per_bldg": v1_avg,
                              "n_active": int(n_cooled), "adoption": np.nan},
        "V2_full_n19063":   {"total_GWh": v2_total_GWh, "avg_kWh_per_bldg": v2_avg,
                              "n_active": int(n_total), "adoption": np.nan},
        **v3_results
    }


def get_per_class_intensity(cdd_value):
    """Compute mean intensity per energy class for cooled stock at given CDD."""
    intensity = beta * cdd_value
    out = {}
    for cls in ["A4", "A3", "A2", "A1", "B", "C", "D", "E", "F", "G"]:
        mask = is_cooled & (energy_class == cls)
        if mask.sum() == 0:
            continue
        total_kWh = (intensity[mask] * area[mask]).sum()
        out[cls] = {
            "n_buildings": int(mask.sum()),
            "total_GWh": total_kWh / 1e6,
            "mean_intensity_kWh_m2": intensity[mask].mean(),
        }
    return out


# ===========================================================================
# STEP 3: COMPUTE PER-MODEL PER-SCENARIO DEMAND
# ===========================================================================

print("\n[2] Computing per-model demand for 4 future scenarios x 12 models...")

per_model_rows = []  # detailed: model, scenario, variant, total_GWh

for scenario, period, y_start, y_end, scenario_label in FUTURE_SCENARIOS:
    # Get the per-model CDD values for this scenario-period
    sc_subset = cdd_table[
        (cdd_table["scenario"] == scenario) &
        (cdd_table["period_start"] == y_start)
    ]
    if len(sc_subset) == 0:
        print(f"  WARN: no CDD data for {scenario_label}")
        continue
    row = sc_subset.iloc[0]

    # Iterate 12 models
    for model in MODELS:
        col_name = f"cdd_{model}"
        if col_name not in row.index or pd.isna(row[col_name]):
            print(f"  WARN: missing CDD for {model} {scenario_label}")
            continue
        cdd_val = float(row[col_name])

        # Compute variants at this CDD
        variants = compute_variants_for_cdd(cdd_val)

        for vname, v in variants.items():
            per_model_rows.append({
                "scenario": scenario_label,
                "scenario_short": f"{scenario}_{period}",
                "period_start": y_start,
                "period_end": y_end,
                "model": model,
                "cdd": round(cdd_val, 1),
                "variant": vname,
                "n_active": v["n_active"],
                "total_GWh": round(v["total_GWh"], 3),
                "avg_kWh_per_bldg": round(v["avg_kWh_per_bldg"], 1),
                "adoption": v["adoption"],
            })

per_model_df = pd.DataFrame(per_model_rows)
per_model_path = TABLES_DIR / "tab_4_2_demand_per_model.csv"
per_model_df.to_csv(per_model_path, index=False)
print(f"  Per-model rows: {len(per_model_df):,}")
print(f"  Saved: {per_model_path.name}")
exec_log.append(f"Per-model demand: {len(per_model_df)} rows saved")

# ===========================================================================
# STEP 4: ENSEMBLE STATISTICS ACROSS 12 MODELS
# ===========================================================================

print("\n[3] Computing ensemble statistics (median, p5, p95) per scenario per variant...")

summary_rows = []

# First the future scenarios with multi-model ensemble
for scenario, period, y_start, y_end, scenario_label in FUTURE_SCENARIOS:
    for vname in ["V1_cooled_n13787", "V2_full_n19063",
                  "V3_Conservative_72pct", "V3_Moderate_85pct", "V3_High_100pct"]:
        subset = per_model_df[
            (per_model_df["scenario"] == scenario_label) &
            (per_model_df["variant"] == vname)
        ]
        if len(subset) == 0:
            continue
        vals = subset["total_GWh"].values
        n_active = int(subset["n_active"].iloc[0])
        avg_per_bldg = float(subset["avg_kWh_per_bldg"].mean())

        summary_rows.append({
            "scenario": scenario_label,
            "variant": vname,
            "anchor_type": "12-model ensemble",
            "n_active": n_active,
            "n_models": len(vals),
            "total_GWh_median": round(np.median(vals), 2),
            "total_GWh_mean": round(np.mean(vals), 2),
            "total_GWh_p05": round(np.percentile(vals, 5), 2),
            "total_GWh_p25": round(np.percentile(vals, 25), 2),
            "total_GWh_p75": round(np.percentile(vals, 75), 2),
            "total_GWh_p95": round(np.percentile(vals, 95), 2),
            "total_GWh_min": round(np.min(vals), 2),
            "total_GWh_max": round(np.max(vals), 2),
            "total_GWh_std": round(np.std(vals), 2),
            "avg_kWh_per_bldg": round(avg_per_bldg, 1),
            "adoption": subset["adoption"].iloc[0],
        })

# Then the ERA5 deterministic scenarios (no ensemble spread)
for scenario_label, cdd_val in ERA5_SCENARIOS:
    variants = compute_variants_for_cdd(cdd_val)
    for vname, v in variants.items():
        summary_rows.append({
            "scenario": scenario_label,
            "variant": vname,
            "anchor_type": "ERA5 observation",
            "n_active": v["n_active"],
            "n_models": 1,
            "total_GWh_median": round(v["total_GWh"], 2),
            "total_GWh_mean": round(v["total_GWh"], 2),
            "total_GWh_p05": round(v["total_GWh"], 2),
            "total_GWh_p25": round(v["total_GWh"], 2),
            "total_GWh_p75": round(v["total_GWh"], 2),
            "total_GWh_p95": round(v["total_GWh"], 2),
            "total_GWh_min": round(v["total_GWh"], 2),
            "total_GWh_max": round(v["total_GWh"], 2),
            "total_GWh_std": 0.0,
            "avg_kWh_per_bldg": round(v["avg_kWh_per_bldg"], 1),
            "adoption": v["adoption"],
        })

summary_df = pd.DataFrame(summary_rows)

# Compute % change vs ERA5 anchors for V1 only (headline narrative)
def pct_change(rows, ref_scenario):
    out = []
    for _, r in rows.iterrows():
        ref = rows.loc[
            (rows["scenario"] == ref_scenario) & (rows["variant"] == r["variant"]),
            "total_GWh_median"
        ]
        if len(ref) > 0 and ref.values[0] > 0:
            out.append(round((r["total_GWh_median"] - ref.values[0]) / ref.values[0] * 100, 1))
        else:
            out.append(np.nan)
    return out

summary_df["pct_vs_1990_2024"] = pct_change(summary_df, "ERA5 historical 1990-2024")
summary_df["pct_vs_2015_2024"] = pct_change(summary_df, "ERA5 recent 2015-2024")

# Reorder: ERA5 first, then ssp245 near, ssp245 far, ssp585 near, ssp585 far
sc_order = [
    "ERA5 historical 1990-2024", "ERA5 recent 2015-2024",
    "SSP2-4.5 / 2030-2050", "SSP2-4.5 / 2080-2100",
    "SSP5-8.5 / 2030-2050", "SSP5-8.5 / 2080-2100",
]
v_order = ["V1_cooled_n13787", "V2_full_n19063",
           "V3_Conservative_72pct", "V3_Moderate_85pct", "V3_High_100pct"]
summary_df["sc_rank"] = summary_df["scenario"].map({s: i for i, s in enumerate(sc_order)})
summary_df["v_rank"] = summary_df["variant"].map({v: i for i, v in enumerate(v_order)})
summary_df = summary_df.sort_values(["sc_rank", "v_rank"]).drop(columns=["sc_rank", "v_rank"]).reset_index(drop=True)

summary_path = TABLES_DIR / "tab_4_2_demand_summary_multimodel.csv"
summary_df.to_csv(summary_path, index=False)
print(f"  Summary rows: {len(summary_df)}")
print(f"  Saved: {summary_path.name}")
exec_log.append(f"Ensemble summary: {len(summary_df)} rows saved")

# ===========================================================================
# STEP 5: PER-CLASS TABLE (cooled stock, ensemble median CDD)
# ===========================================================================

print("\n[4] Computing per-class intensity table (cooled stock, ensemble median)...")

class_rows = []

# ERA5 anchors
for scenario_label, cdd_val in ERA5_SCENARIOS:
    per_cls = get_per_class_intensity(cdd_val)
    for cls, d in per_cls.items():
        class_rows.append({
            "scenario": scenario_label, "anchor_type": "ERA5",
            "energy_class": cls, "n_buildings": d["n_buildings"],
            "total_GWh": round(d["total_GWh"], 3),
            "mean_intensity_kWh_m2": round(d["mean_intensity_kWh_m2"], 2),
        })

# Future scenarios use the ensemble median CDD
for scenario, period, y_start, y_end, scenario_label in FUTURE_SCENARIOS:
    sc_subset = cdd_table[
        (cdd_table["scenario"] == scenario) &
        (cdd_table["period_start"] == y_start)
    ]
    if len(sc_subset) == 0:
        continue
    cdd_val = float(sc_subset["cdd_ensemble_median"].iloc[0])

    per_cls = get_per_class_intensity(cdd_val)
    for cls, d in per_cls.items():
        class_rows.append({
            "scenario": scenario_label, "anchor_type": "ensemble median",
            "energy_class": cls, "n_buildings": d["n_buildings"],
            "total_GWh": round(d["total_GWh"], 3),
            "mean_intensity_kWh_m2": round(d["mean_intensity_kWh_m2"], 2),
        })

class_df = pd.DataFrame(class_rows)
class_path = TABLES_DIR / "tab_4_3_demand_per_class_multimodel.csv"
class_df.to_csv(class_path, index=False)
print(f"  Per-class rows: {len(class_df)}")
print(f"  Saved: {class_path.name}")
exec_log.append(f"Per-class table saved")

# ===========================================================================
# STEP 6: HEADLINE COMPARISON (old v2 vs new multimodel)
# ===========================================================================

print("\n" + "=" * 75)
print("HEADLINE COMPARISON (V1 cooled stock)")
print("=" * 75)

# Locked v2 values from your existing methodology
old_v2 = {
    "ERA5 historical 1990-2024": 30.06,   # GWh
    "ERA5 recent 2015-2024":     38.30,
    "SSP2-4.5 / 2030-2050":      None,    # not in old table, will be n/a
    "SSP2-4.5 / 2080-2100":      59.81,
    "SSP5-8.5 / 2030-2050":      None,
    "SSP5-8.5 / 2080-2100":     108.47,
}

print(f"\n  {'Scenario':<32} {'OLD v2 (GWh)':<15} {'NEW multi-model (GWh)':<25}")
print("  " + "-" * 73)
for sc in sc_order:
    new_row = summary_df[
        (summary_df["scenario"] == sc) & (summary_df["variant"] == "V1_cooled_n13787")
    ]
    if len(new_row) == 0:
        continue
    new_med = new_row["total_GWh_median"].iloc[0]
    new_p5 = new_row["total_GWh_p05"].iloc[0]
    new_p95 = new_row["total_GWh_p95"].iloc[0]
    old = old_v2.get(sc)
    if old is None:
        new_str = f"{new_med:.1f} [{new_p5:.1f}-{new_p95:.1f}]"
        print(f"  {sc:<32} {'(not in old)':<15} {new_str:<25}")
    elif new_p5 == new_p95:
        # ERA5 deterministic
        print(f"  {sc:<32} {old:<15.2f} {new_med:<25.2f}")
    else:
        new_str = f"{new_med:.1f} [{new_p5:.1f}-{new_p95:.1f}]"
        print(f"  {sc:<32} {old:<15.2f} {new_str:<25}")

# ===========================================================================
# STEP 7: GENERATE FIGURE 4.3 MULTIMODEL
# ===========================================================================

print("\n[5] Generating Figure 4.3 multi-model...")

# Three-panel figure: V1, V2, V3
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sc_short = {
    "ERA5 historical 1990-2024": "Hist\n1990-2024",
    "ERA5 recent 2015-2024":     "Recent\n2015-2024",
    "SSP2-4.5 / 2030-2050":      "SSP245\n2030-2050",
    "SSP2-4.5 / 2080-2100":      "SSP245\n2080-2100",
    "SSP5-8.5 / 2030-2050":      "SSP585\n2030-2050",
    "SSP5-8.5 / 2080-2100":      "SSP585\n2080-2100",
}
sc_colors_list = ["#333333", "#777777", "#4A7AB8", "#1F4E79", "#D49A4A", "#A03020"]

# Panel (a): V1 cooled stock
ax = axes[0]
v1_data = summary_df[summary_df["variant"] == "V1_cooled_n13787"]
v1_data = v1_data.set_index("scenario").reindex(sc_order).reset_index()
medians = v1_data["total_GWh_median"].values
p5 = v1_data["total_GWh_p05"].values
p95 = v1_data["total_GWh_p95"].values
err_low = medians - p5
err_high = p95 - medians

x = np.arange(len(sc_order))
ax.bar(x, medians, color=sc_colors_list, edgecolor="black", linewidth=0.5,
       yerr=[err_low, err_high], capsize=6, ecolor="black",
       error_kw=dict(linewidth=1.2))
for i, (m, lo, hi) in enumerate(zip(medians, p5, p95)):
    if lo == hi:
        # Deterministic ERA5
        ax.text(i, m + max(medians)*0.02, f"{m:.0f}", ha="center", fontsize=9, fontweight="bold")
    else:
        ax.text(i, hi + max(medians)*0.02, f"{m:.0f}", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([sc_short[s] for s in sc_order], fontsize=8)
ax.set_ylabel("Total cooling demand (GWh/yr)")
ax.set_title(f"(a) V1: cooled stock (n={n_cooled:,}) - headline",
             fontsize=11, fontweight="bold", color=HEADING, loc="left")
ax.grid(axis="y", alpha=0.3, linewidth=0.5)

# Panel (b): V2 full stock
ax = axes[1]
v2_data = summary_df[summary_df["variant"] == "V2_full_n19063"]
v2_data = v2_data.set_index("scenario").reindex(sc_order).reset_index()
medians = v2_data["total_GWh_median"].values
p5 = v2_data["total_GWh_p05"].values
p95 = v2_data["total_GWh_p95"].values
err_low = medians - p5
err_high = p95 - medians

ax.bar(x, medians, color=sc_colors_list, edgecolor="black", linewidth=0.5,
       yerr=[err_low, err_high], capsize=6, ecolor="black",
       error_kw=dict(linewidth=1.2))
for i, (m, lo, hi) in enumerate(zip(medians, p5, p95)):
    if lo == hi:
        ax.text(i, m + max(medians)*0.02, f"{m:.0f}", ha="center", fontsize=9, fontweight="bold")
    else:
        ax.text(i, hi + max(medians)*0.02, f"{m:.0f}", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([sc_short[s] for s in sc_order], fontsize=8)
ax.set_ylabel("Total cooling demand (GWh/yr)")
ax.set_title(f"(b) V2: full EPC stock if all cooled (n={n_total:,}) - upper bound",
             fontsize=11, fontweight="bold", color=HEADING, loc="left")
ax.grid(axis="y", alpha=0.3, linewidth=0.5)

# Panel (c): V3 adoption pathways under SSP5-8.5/2080
ax = axes[2]
v3_data = summary_df[
    (summary_df["scenario"] == "SSP5-8.5 / 2080-2100") &
    (summary_df["variant"].str.startswith("V3_"))
].copy()
v3_data["v_order"] = v3_data["variant"].map(
    {"V3_Conservative_72pct": 0, "V3_Moderate_85pct": 1, "V3_High_100pct": 2}
)
v3_data = v3_data.sort_values("v_order")

v3_medians = v3_data["total_GWh_median"].values
v3_p5 = v3_data["total_GWh_p05"].values
v3_p95 = v3_data["total_GWh_p95"].values
v3_err_low = v3_medians - v3_p5
v3_err_high = v3_p95 - v3_medians

xv = np.arange(3)
ax.bar(xv, v3_medians, color=["#D49A4A", "#A03020", "#5A0000"],
       edgecolor="black", linewidth=0.5,
       yerr=[v3_err_low, v3_err_high], capsize=6, ecolor="black",
       error_kw=dict(linewidth=1.2))
for i, (m, hi) in enumerate(zip(v3_medians, v3_p95)):
    ax.text(i, hi + max(v3_medians)*0.02, f"{m:.0f}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(xv)
ax.set_xticklabels(["72%\n(today)", "85%\n(moderate)", "100%\n(full)"], fontsize=9)
ax.set_ylabel("Total cooling demand (GWh/yr)")
ax.set_title("(c) V3: adoption pathways under SSP5-8.5/2080-2100",
             fontsize=11, fontweight="bold", color=HEADING, loc="left")
ax.grid(axis="y", alpha=0.3, linewidth=0.5)

fig.suptitle("Milan residential cooling demand: 12-model NEX-GDDP-CMIP6 ensemble (multimodel revision)",
             fontsize=13, fontweight="bold", color=HEADING, y=1.02)
plt.tight_layout()

fig_png = FIGURES_DIR / "fig_4_3_demand_evolution_3panel_multimodel.png"
fig_pdf = FIGURES_DIR / "fig_4_3_demand_evolution_3panel_multimodel.pdf"
plt.savefig(fig_png, dpi=300, bbox_inches="tight")
plt.savefig(fig_pdf, bbox_inches="tight")
plt.close()

# Caption
caption_path = FIGURES_DIR / "fig_4_3_demand_evolution_3panel_multimodel.caption.txt"
with open(caption_path, "w", encoding="utf-8") as f:
    f.write(
        "Figure 4.3 (multimodel). Milan residential cooling demand under three projection "
        "variants and six climate scenarios. Bars show ensemble median across 12 CMIP6 "
        "models for future scenarios; error bars show 5th-95th percentile range. ERA5 "
        "observation values shown as deterministic anchors (no error bars). Variants: "
        "(a) V1 - currently cooled stock (n=13,787 buildings, headline); (b) V2 - full "
        "EPC stock if all buildings adopted cooling (n=19,063, technical upper bound); "
        "(c) V3 - three adoption pathways under SSP5-8.5/2080-2100 showing how future "
        "demand depends on AC adoption rate."
    )
print(f"  Saved: {fig_png.name}")
print(f"  Saved: {caption_path.name}")
exec_log.append(f"Figure 4.3 multimodel saved")

# ===========================================================================
# STEP 8: WRITE LOGS
# ===========================================================================

exec_log.extend([
    "",
    "=" * 75,
    f"Script B finished: {datetime.now().isoformat()}",
])
exec_log_path = LOGS_DIR / "script_b_execution_log.txt"
with open(exec_log_path, "w", encoding="utf-8") as f:
    f.write("\n".join(exec_log))

# ===========================================================================
# FINAL SUMMARY
# ===========================================================================

print("\n" + "=" * 75)
print("SCRIPT B COMPLETE")
print("=" * 75)
print(f"\nOutputs in: {MULTIMODEL_ROOT}")
print(f"\nKey deliverables:")
print(f"  Tables:")
print(f"    {summary_path.name}      (main ensemble summary)")
print(f"    {per_model_path.name}    (full per-model detail)")
print(f"    {class_path.name}        (per-class breakdown)")
print(f"  Figures:")
print(f"    {fig_png.name}")
print(f"\nReady for Script C (uncertainty quantification with model spread).")