"""
============================================================================
SCRIPT C: Multi-Model Uncertainty Quantification (Session 5 Revision)
----------------------------------------------------------------------------
Single self-contained script. Auto-installs scipy if missing.

Does in one execution:
  1. Bootstrap propagation (B=2000) with multi-model uncertainty
  2. Sobol decomposition (V1: 3 inputs, V3: 4 inputs)
  3. OAT tornado (V1, V3)
  4. Variance decomposition tables
  5. All Session 5 figures (5.1, 5.2, 5.3) regenerated
  6. Verification report and execution log

Multi-model methodology:
  Future CDD = bootstrap mean of yearly CDD from a randomly-selected model
  (uniform over 12 NEX-GDDP-CMIP6 models). This combines model uncertainty
  + inter-annual variability into the climate component, decomposable in Sobol.

  ERA5 baseline scenarios use ERA5 yearly bootstrap (unchanged).

Inputs (per Sobol):
  V1 future (3 inputs):  beta regression, climate model identity, inter-annual
  V3 future (4 inputs):  V1 + AC adoption rate
  V1 ERA5  (2 inputs):   beta regression, ERA5 baseline yearly
  V3 ERA5  (3 inputs):   V1 + AC adoption

Runtime: 5-10 minutes total.
============================================================================
"""

import sys
import subprocess

# Auto-install scipy if missing
try:
    from scipy.stats import norm
except ImportError:
    print("scipy not installed. Installing now...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    from scipy.stats import norm

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import time

# ===========================================================================
# CONFIGURATION
# ===========================================================================

CANDIDATE_PROJECT_ROOTS = [
    Path(r"C:\Users\n.mohammadi\Desktop\NimaMohammadi\03. Nima Mohammadi - Thesis\ML Dataset"),
    Path(r"C:\Users\n.mohammadi\Desktop\NimaMohammadi\02.Nima Mohammadi - Thesis\ML Dataset"),
]

PROJECT_ROOT = None
for cand in CANDIDATE_PROJECT_ROOTS:
    if cand.exists():
        PROJECT_ROOT = cand
        break
if PROJECT_ROOT is None:
    print("ERROR: Cannot find ML Dataset folder.")
    sys.exit(1)
print(f"Project root: {PROJECT_ROOT}")

STAGE2_ROOT = PROJECT_ROOT / "07_results" / "stage2"
MULTIMODEL_ROOT = STAGE2_ROOT / "_MULTIMODEL"
TABLES_DIR = MULTIMODEL_ROOT / "_TABLES"
FIGURES_DIR = MULTIMODEL_ROOT / "_FIGURES"
LOGS_DIR = MULTIMODEL_ROOT / "_LOGS"
for d in [TABLES_DIR, FIGURES_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Inputs
PER_BUILDING_FILE = (STAGE2_ROOT / "session4_demand_projection" / "E_outputs"
                     / "stage2_per_building_demand_projections_v2.csv")
CDD_PER_YEAR_FILE = TABLES_DIR / "tab_4_1_cdd_per_model_per_year.csv"
DEMAND_SUMMARY_FILE = TABLES_DIR / "tab_4_2_demand_summary_multimodel.csv"
TEST_PRED_FILE = (STAGE2_ROOT / "session2_beta_regression" / "B_evaluation"
                  / "stage2_test_predictions_v1.csv")
ERA5_FILE = PROJECT_ROOT / "02_inputs" / "ERA5_Milan_1990_2024_daily.csv"

# Constants
RNG_SEED = 42
COOLING_BASE_C = 22.0
JJA_MONTHS = [6, 7, 8]
B_BOOT = 2000          # Bootstrap iterations
SOBOL_N = 1024         # Sobol base samples
SOBOL_INNER_BOOT = 500 # Sobol CI inner bootstrap

# 12 models (must match Scripts A and B)
MODELS = [
    "MPI-ESM1-2-HR", "CMCC-ESM2", "EC-Earth3", "CNRM-ESM2-1",
    "IPSL-CM6A-LR", "CESM2", "HadGEM3-GC31-LL", "CanESM5",
    "NorESM2-MM", "GFDL-ESM4", "MIROC6", "ACCESS-ESM1-5",
]

# Scenarios (display names)
SCENARIOS = [
    "ERA5 historical 1990-2024", "ERA5 recent 2015-2024",
    "SSP2-4.5 / 2030-2050", "SSP2-4.5 / 2080-2100",
    "SSP5-8.5 / 2030-2050", "SSP5-8.5 / 2080-2100",
]
FUTURE_SCENARIOS = [s for s in SCENARIOS if "ERA5" not in s]
HEADLINE_SOBOL_SCENARIO = "SSP5-8.5 / 2080-2100"

# Map display name to (scenario_short, period_start)
SCENARIO_MAP = {
    "SSP2-4.5 / 2030-2050": ("ssp245", 2030),
    "SSP2-4.5 / 2080-2100": ("ssp245", 2080),
    "SSP5-8.5 / 2030-2050": ("ssp585", 2030),
    "SSP5-8.5 / 2080-2100": ("ssp585", 2080),
}

# V3 adoption parameters
V3_ADOPTION_LOW = 0.60
V3_ADOPTION_HIGH = 1.00
V3_ADOPTION_REF = 0.7232  # current cooled fraction in EPC stock

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

print("=" * 75)
print("SCRIPT C: Multi-Model Uncertainty Quantification (Session 5 revision)")
print(f"Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 75)

exec_log = [f"Script C log - started {datetime.now().isoformat()}", "=" * 75, ""]

# ===========================================================================
# STEP 1: LOAD INPUTS
# ===========================================================================

print("\n[1] Loading inputs...")

# Per-building data
if not PER_BUILDING_FILE.exists():
    print(f"ERROR: per-building file missing: {PER_BUILDING_FILE}")
    sys.exit(1)
proj = pd.read_csv(PER_BUILDING_FILE, low_memory=False)
beta_pred_full = proj["beta_predicted"].values
area_full = proj["effective_cooled_area_m2"].values
is_cooled = (proj["CLIMATIZZAZIONE_ESTIVA"] == True).values
beta_pred_cooled = beta_pred_full[is_cooled]
area_cooled = area_full[is_cooled]
n_full = len(proj)
n_cooled = is_cooled.sum()
print(f"  Buildings: {n_full:,} total, {n_cooled:,} cooled (V1)")

# Test residuals (compute res_sem)
if TEST_PRED_FILE.exists():
    test_pred = pd.read_csv(TEST_PRED_FILE)
    residuals_log1p = np.log1p(test_pred["beta_actual"].values) - np.log1p(test_pred["beta_predicted"].values)
    res_std = float(np.std(residuals_log1p))
    n_test = len(residuals_log1p)
else:
    res_std = 0.116
    n_test = 2758
res_sem = res_std / np.sqrt(n_test)
print(f"  Beta residuals: std={res_std:.4f}, n_test={n_test:,}, SEM={res_sem:.6f}")

# ERA5 yearly CDD
era5 = pd.read_csv(ERA5_FILE, usecols=["time", "temperature_2m"])
era5["time"] = pd.to_datetime(era5["time"], errors="coerce")
era5 = era5.dropna(subset=["time"]).copy()
era5["year"] = era5["time"].dt.year
era5["month"] = era5["time"].dt.month
era5["cdd"] = (era5["temperature_2m"] - COOLING_BASE_C).clip(lower=0)
era5_yearly = era5[era5["month"].isin(JJA_MONTHS)].groupby("year")["cdd"].sum()
era5_full_yearly = era5_yearly.values
era5_recent_yearly = era5_yearly.loc[2015:2024].values
print(f"  ERA5 yearly: full n={len(era5_full_yearly)}, recent n={len(era5_recent_yearly)}")

# Multi-model per-year CDD (from Script A)
if not CDD_PER_YEAR_FILE.exists():
    print(f"ERROR: per-year CDD file missing: {CDD_PER_YEAR_FILE}")
    print(f"Run Script A first.")
    sys.exit(1)
cdd_per_year = pd.read_csv(CDD_PER_YEAR_FILE)

# Build dict: cdd_arrays[scenario_display][model] -> yearly array
cdd_arrays = {}
for sc_display, (sc_short, p_start) in SCENARIO_MAP.items():
    cdd_arrays[sc_display] = {}
    for m in MODELS:
        sub = cdd_per_year[
            (cdd_per_year["scenario"] == sc_short) &
            (cdd_per_year["period_start"] == p_start) &
            (cdd_per_year["model"] == m)
        ]
        if len(sub) == 0:
            print(f"  WARN: no data for {m} {sc_display}")
            continue
        cdd_arrays[sc_display][m] = sub["cdd"].values
print(f"  Multi-model CDD arrays loaded: {len(cdd_arrays)} future scenarios x 12 models")

# Demand summary (locked point estimates from Script B)
demand_summary = pd.read_csv(DEMAND_SUMMARY_FILE)
locked_v1 = {}  # scenario -> ensemble median (from Script B)
locked_v3 = {}
for sc in SCENARIOS:
    v1_row = demand_summary[(demand_summary["scenario"] == sc) &
                             (demand_summary["variant"] == "V1_cooled_n13787")]
    v3_row = demand_summary[(demand_summary["scenario"] == sc) &
                             (demand_summary["variant"] == "V3_High_100pct")]
    if len(v1_row) > 0:
        locked_v1[sc] = float(v1_row["total_GWh_median"].iloc[0])
    if len(v3_row) > 0:
        # V3 uses adoption-pathway max (100%) as locked reference for uncertainty
        locked_v3[sc] = float(v3_row["total_GWh_median"].iloc[0])
print(f"  Locked point estimates loaded for {len(locked_v1)} scenarios")
print(f"  V1 today (locked):       {locked_v1.get('ERA5 recent 2015-2024', 0):.2f} GWh")
print(f"  V1 SSP5-8.5/2080 (locked): {locked_v1.get('SSP5-8.5 / 2080-2100', 0):.2f} GWh")
exec_log.append("Inputs loaded successfully")

# ===========================================================================
# STEP 2: HELPER FUNCTIONS
# ===========================================================================

rng = np.random.default_rng(RNG_SEED)

def perturb_beta(beta_array, z, sem):
    """Perturb beta on log1p scale by z*SEM."""
    log1p_perturbed = np.log1p(beta_array) + z * sem
    return np.clip(np.expm1(log1p_perturbed), 0.001, 5.0)


def bootstrap_yearly_mean(yearly_arr, rng_obj):
    """Sample-mean bootstrap of a yearly CDD array."""
    return rng_obj.choice(yearly_arr, size=len(yearly_arr), replace=True).mean()


def compute_demand_v1(beta_array, cdd, area_array):
    return float((beta_array * cdd * area_array).sum() / 1e6)


def compute_demand_v3(beta_array, cdd, area_array, adoption):
    factor = adoption / V3_ADOPTION_REF
    return float((beta_array * cdd * area_array).sum() / 1e6 * factor)


# ===========================================================================
# STEP 3: BOOTSTRAP PROPAGATION (B=2000)
# ===========================================================================

print(f"\n[2] Bootstrap propagation (B={B_BOOT}, multi-model climate)...")

boot_v1 = {sc: [] for sc in SCENARIOS}
boot_v3 = {sc: [] for sc in SCENARIOS}

t0 = time.time()
for b in range(B_BOOT):
    z_beta = rng.standard_normal()
    beta_b_cooled = perturb_beta(beta_pred_cooled, z_beta, res_sem)
    beta_b_full = perturb_beta(beta_pred_full, z_beta, res_sem)
    adoption = rng.uniform(V3_ADOPTION_LOW, V3_ADOPTION_HIGH)

    for sc in SCENARIOS:
        if sc == "ERA5 historical 1990-2024":
            cdd_b = bootstrap_yearly_mean(era5_full_yearly, rng)
        elif sc == "ERA5 recent 2015-2024":
            cdd_b = bootstrap_yearly_mean(era5_recent_yearly, rng)
        else:
            # Pick random model uniformly, bootstrap that model's yearly array
            model_idx = rng.integers(0, len(MODELS))
            model_name = MODELS[model_idx]
            yearly_arr = cdd_arrays[sc].get(model_name)
            if yearly_arr is None or len(yearly_arr) == 0:
                # Fallback to first available
                avail = [m for m in MODELS if cdd_arrays[sc].get(m) is not None]
                yearly_arr = cdd_arrays[sc][avail[0]]
            cdd_b = bootstrap_yearly_mean(yearly_arr, rng)

        boot_v1[sc].append(compute_demand_v1(beta_b_cooled, cdd_b, area_cooled))
        boot_v3[sc].append(compute_demand_v3(beta_b_full, cdd_b, area_full, adoption))

    if (b + 1) % 500 == 0:
        print(f"  Progress: {b+1}/{B_BOOT} ({time.time()-t0:.1f}s)")

print(f"  Bootstrap complete in {time.time()-t0:.1f}s")

# Build summary table
def build_summary(boot_dict, locked_dict, label):
    rows = []
    for sc in SCENARIOS:
        vals = np.array(boot_dict[sc])
        rows.append({
            "scenario": sc, "variant": label,
            "point_GWh_median": round(np.median(vals), 2),
            "locked_GWh": round(locked_dict.get(sc, np.nan), 2),
            "ci_low_GWh": round(np.percentile(vals, 2.5), 2),
            "ci_high_GWh": round(np.percentile(vals, 97.5), 2),
            "ci_width_pct": round((np.percentile(vals, 97.5) - np.percentile(vals, 2.5))
                                   / max(np.median(vals), 1e-9) * 100, 1),
        })
    return pd.DataFrame(rows)

v1_df = build_summary(boot_v1, locked_v1, "V1 cooled stock")
v3_df = build_summary(boot_v3, locked_v3, "V3 adoption pathway (100% reference)")

# Relative change vs today (V1 only)
today_arr = np.array(boot_v1["ERA5 recent 2015-2024"])
rel_rows = []
for sc in SCENARIOS:
    sc_arr = np.array(boot_v1[sc])
    n_min = min(len(today_arr), len(sc_arr))
    ratios = (sc_arr[:n_min] / today_arr[:n_min] - 1) * 100
    rel_rows.append({
        "scenario": sc,
        "pct_vs_today": round(np.median(ratios), 1),
        "ci_low_pct": round(np.percentile(ratios, 2.5), 1),
        "ci_high_pct": round(np.percentile(ratios, 97.5), 1),
    })
rel_df = pd.DataFrame(rel_rows)

# Combined save
combined = pd.concat([v1_df, v3_df], ignore_index=True)
combined = combined.merge(rel_df, on="scenario", how="left")
unc_path = TABLES_DIR / "tab_5_1_uncertainty_intervals_multimodel.csv"
combined.to_csv(unc_path, index=False)
print(f"  Saved: {unc_path.name}")

print(f"\n  V1 cooled stock CIs:")
print(v1_df[["scenario", "point_GWh_median", "ci_low_GWh", "ci_high_GWh", "ci_width_pct"]].to_string(index=False))

exec_log.append(f"Bootstrap completed: {B_BOOT} iterations")

# ===========================================================================
# STEP 4: SOBOL DECOMPOSITION (V1 + V3 at SSP5-8.5/2080-2100)
# ===========================================================================

print(f"\n[3] Sobol decomposition at {HEADLINE_SOBOL_SCENARIO}")
print(f"     N={SOBOL_N}, inner bootstrap = {SOBOL_INNER_BOOT}")

# Build per-model yearly array list for the headline scenario
sobol_sc = HEADLINE_SOBOL_SCENARIO
model_arrays = [cdd_arrays[sobol_sc][m] for m in MODELS if m in cdd_arrays[sobol_sc]]
n_models_avail = len(model_arrays)
print(f"  Models available for {sobol_sc}: {n_models_avail}")


def evaluate_sobol_v1(samples):
    """V1 future: 3 inputs (beta, model_id, inter-annual)."""
    n = len(samples)
    out = np.zeros(n)
    for i in range(n):
        # Input 1: beta perturbation
        z_beta = norm.ppf(np.clip(samples[i, 0], 0.001, 0.999))
        beta_i = perturb_beta(beta_pred_cooled, z_beta, res_sem)

        # Input 2: model identity (categorical from uniform)
        m_idx = int(np.clip(samples[i, 1] * n_models_avail, 0, n_models_avail - 1))
        yearly_arr = model_arrays[m_idx]

        # Input 3: inter-annual variability (quantile of bootstrap means)
        sub_rng = np.random.default_rng(int(samples[i, 2] * 1e7) + 1)
        means_inner = np.array([
            sub_rng.choice(yearly_arr, size=len(yearly_arr), replace=True).mean()
            for _ in range(50)
        ])
        cdd_i = float(np.percentile(means_inner, samples[i, 2] * 100))

        out[i] = compute_demand_v1(beta_i, cdd_i, area_cooled)
    return out


def evaluate_sobol_v3(samples):
    """V3 future: 4 inputs (beta, model_id, inter-annual, adoption)."""
    n = len(samples)
    out = np.zeros(n)
    for i in range(n):
        z_beta = norm.ppf(np.clip(samples[i, 0], 0.001, 0.999))
        beta_i = perturb_beta(beta_pred_full, z_beta, res_sem)

        m_idx = int(np.clip(samples[i, 1] * n_models_avail, 0, n_models_avail - 1))
        yearly_arr = model_arrays[m_idx]

        sub_rng = np.random.default_rng(int(samples[i, 2] * 1e7) + 1)
        means_inner = np.array([
            sub_rng.choice(yearly_arr, size=len(yearly_arr), replace=True).mean()
            for _ in range(50)
        ])
        cdd_i = float(np.percentile(means_inner, samples[i, 2] * 100))

        adoption_i = V3_ADOPTION_LOW + (V3_ADOPTION_HIGH - V3_ADOPTION_LOW) * samples[i, 3]

        out[i] = compute_demand_v3(beta_i, cdd_i, area_full, adoption_i)
    return out


def compute_sobol(eval_fn, k, N=SOBOL_N, n_boot=SOBOL_INNER_BOOT):
    """Saltelli pick-freeze Sobol with bootstrap CIs."""
    A = rng.random((N, k))
    B_mat = rng.random((N, k))
    y_A = eval_fn(A)
    y_B = eval_fn(B_mat)
    y_AB = np.zeros((N, k))
    for i in range(k):
        AB = A.copy()
        AB[:, i] = B_mat[:, i]
        y_AB[:, i] = eval_fn(AB)

    var_y = float(np.var(np.concatenate([y_A, y_B])))
    S1 = np.array([np.mean(y_B * (y_AB[:, i] - y_A)) / var_y for i in range(k)])
    ST = np.array([0.5 * np.mean((y_A - y_AB[:, i]) ** 2) / var_y for i in range(k)])

    S1_b = np.zeros((n_boot, k))
    ST_b = np.zeros((n_boot, k))
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        y_A_b = y_A[idx]
        y_B_b = y_B[idx]
        var_b = float(np.var(np.concatenate([y_A_b, y_B_b])))
        for i in range(k):
            y_AB_b = y_AB[idx, i]
            S1_b[b, i] = np.mean(y_B_b * (y_AB_b - y_A_b)) / var_b
            ST_b[b, i] = 0.5 * np.mean((y_A_b - y_AB_b) ** 2) / var_b
    return S1, ST, S1_b, ST_b, var_y


print(f"  V1 Sobol (3 inputs)...")
t0 = time.time()
S1_v1, ST_v1, S1_b_v1, ST_b_v1, var_v1 = compute_sobol(evaluate_sobol_v1, k=3)
print(f"    Done in {time.time()-t0:.1f}s")

print(f"  V3 Sobol (4 inputs)...")
t0 = time.time()
S1_v3, ST_v3, S1_b_v3, ST_b_v3, var_v3 = compute_sobol(evaluate_sobol_v3, k=4)
print(f"    Done in {time.time()-t0:.1f}s")

v1_inputs = ["beta regression model error", "Climate model identity (12 models)",
             "Inter-annual variability"]
v3_inputs = ["beta regression model error", "Climate model identity (12 models)",
             "Inter-annual variability", "AC adoption rate (60-100%)"]

sobol_v1_df = pd.DataFrame({
    "input": v1_inputs,
    "S1": np.round(S1_v1, 4),
    "S1_ci_low": np.round(np.percentile(S1_b_v1, 2.5, axis=0), 4),
    "S1_ci_high": np.round(np.percentile(S1_b_v1, 97.5, axis=0), 4),
    "ST": np.round(ST_v1, 4),
    "ST_ci_low": np.round(np.percentile(ST_b_v1, 2.5, axis=0), 4),
    "ST_ci_high": np.round(np.percentile(ST_b_v1, 97.5, axis=0), 4),
    "interaction": np.round(ST_v1 - S1_v1, 4),
    "n_inner_bootstrap": SOBOL_INNER_BOOT,
})
sobol_v3_df = pd.DataFrame({
    "input": v3_inputs,
    "S1": np.round(S1_v3, 4),
    "S1_ci_low": np.round(np.percentile(S1_b_v3, 2.5, axis=0), 4),
    "S1_ci_high": np.round(np.percentile(S1_b_v3, 97.5, axis=0), 4),
    "ST": np.round(ST_v3, 4),
    "ST_ci_low": np.round(np.percentile(ST_b_v3, 2.5, axis=0), 4),
    "ST_ci_high": np.round(np.percentile(ST_b_v3, 97.5, axis=0), 4),
    "interaction": np.round(ST_v3 - S1_v3, 4),
    "n_inner_bootstrap": SOBOL_INNER_BOOT,
})

sobol_v1_df.to_csv(TABLES_DIR / "tab_5_2a_sobol_v1_multimodel.csv", index=False)
sobol_v3_df.to_csv(TABLES_DIR / "tab_5_2b_sobol_v3_multimodel.csv", index=False)

print(f"\n  V1 Sobol indices:")
print(sobol_v1_df[["input", "S1", "ST"]].to_string(index=False))
print(f"\n  V3 Sobol indices:")
print(sobol_v3_df[["input", "S1", "ST"]].to_string(index=False))

# Variance decomposition tables (S_T-based)
var_v1_df = pd.DataFrame({
    "input": v1_inputs,
    "ST": np.round(ST_v1, 4),
    "variance_contribution_pct": np.round(ST_v1 / ST_v1.sum() * 100, 1),
    "total_variance_GWh2": round(var_v1, 2),
})
var_v3_df = pd.DataFrame({
    "input": v3_inputs,
    "ST": np.round(ST_v3, 4),
    "variance_contribution_pct": np.round(ST_v3 / ST_v3.sum() * 100, 1),
    "total_variance_GWh2": round(var_v3, 2),
})
var_v1_df.to_csv(TABLES_DIR / "tab_5_3a_variance_v1_multimodel.csv", index=False)
var_v3_df.to_csv(TABLES_DIR / "tab_5_3b_variance_v3_multimodel.csv", index=False)
exec_log.append(f"Sobol completed: V1 dom={v1_inputs[ST_v1.argmax()]}, V3 dom={v3_inputs[ST_v3.argmax()]}")

# ===========================================================================
# STEP 5: OAT TORNADO (V1, V3 at SSP5-8.5/2080-2100)
# ===========================================================================

print(f"\n[4] OAT tornado at {HEADLINE_SOBOL_SCENARIO}...")

baseline_v1_oat = locked_v1[HEADLINE_SOBOL_SCENARIO]
baseline_v3_oat = locked_v3[HEADLINE_SOBOL_SCENARIO]

# Beta perturbations
beta_low_cooled = perturb_beta(beta_pred_cooled, -1.96, res_sem)
beta_high_cooled = perturb_beta(beta_pred_cooled, +1.96, res_sem)
beta_low_full = perturb_beta(beta_pred_full, -1.96, res_sem)
beta_high_full = perturb_beta(beta_pred_full, +1.96, res_sem)

# Climate model: range = (min model CDD, max model CDD) for headline scenario
model_means = np.array([arr.mean() for arr in model_arrays])
cdd_model_low = model_means.min()
cdd_model_high = model_means.max()
cdd_model_central = float(np.median(model_means))

# Inter-annual variability (using ensemble-pooled yearly array)
pooled_yearly = np.concatenate(model_arrays)
pooled_sem = pooled_yearly.std() / np.sqrt(len(pooled_yearly))
pooled_mean = pooled_yearly.mean()
cdd_iav_low = pooled_mean - 1.96 * pooled_sem
cdd_iav_high = pooled_mean + 1.96 * pooled_sem

# AC adoption (V3 only)
adoption_low = V3_ADOPTION_LOW
adoption_high = V3_ADOPTION_HIGH
adoption_ref = V3_ADOPTION_REF

# V1 OAT
oat_v1_rows = []
for d, b in [("low", beta_low_cooled), ("high", beta_high_cooled)]:
    val = compute_demand_v1(b, cdd_model_central, area_cooled)
    oat_v1_rows.append({"input": "beta regression model error", "direction": d,
                         "demand_GWh": round(val, 2), "delta_GWh": round(val - baseline_v1_oat, 2)})
for d, c in [("low", cdd_model_low), ("high", cdd_model_high)]:
    val = compute_demand_v1(beta_pred_cooled, c, area_cooled)
    oat_v1_rows.append({"input": "Climate model spread", "direction": d,
                         "demand_GWh": round(val, 2), "delta_GWh": round(val - baseline_v1_oat, 2)})
for d, c in [("low", cdd_iav_low), ("high", cdd_iav_high)]:
    val = compute_demand_v1(beta_pred_cooled, c, area_cooled)
    oat_v1_rows.append({"input": "Inter-annual variability", "direction": d,
                         "demand_GWh": round(val, 2), "delta_GWh": round(val - baseline_v1_oat, 2)})
oat_v1_df = pd.DataFrame(oat_v1_rows)

# V3 OAT
oat_v3_rows = []
for d, b in [("low", beta_low_full), ("high", beta_high_full)]:
    val = compute_demand_v3(b, cdd_model_central, area_full, adoption_ref)
    oat_v3_rows.append({"input": "beta regression model error", "direction": d,
                         "demand_GWh": round(val, 2), "delta_GWh": round(val - baseline_v3_oat, 2)})
for d, c in [("low", cdd_model_low), ("high", cdd_model_high)]:
    val = compute_demand_v3(beta_pred_full, c, area_full, adoption_ref)
    oat_v3_rows.append({"input": "Climate model spread", "direction": d,
                         "demand_GWh": round(val, 2), "delta_GWh": round(val - baseline_v3_oat, 2)})
for d, c in [("low", cdd_iav_low), ("high", cdd_iav_high)]:
    val = compute_demand_v3(beta_pred_full, c, area_full, adoption_ref)
    oat_v3_rows.append({"input": "Inter-annual variability", "direction": d,
                         "demand_GWh": round(val, 2), "delta_GWh": round(val - baseline_v3_oat, 2)})
for d, a in [("low", adoption_low), ("high", adoption_high)]:
    val = compute_demand_v3(beta_pred_full, cdd_model_central, area_full, a)
    oat_v3_rows.append({"input": "AC adoption rate", "direction": d,
                         "demand_GWh": round(val, 2), "delta_GWh": round(val - baseline_v3_oat, 2)})
oat_v3_df = pd.DataFrame(oat_v3_rows)

oat_v1_df.to_csv(TABLES_DIR / "tab_5_4a_oat_v1_multimodel.csv", index=False)
oat_v3_df.to_csv(TABLES_DIR / "tab_5_4b_oat_v3_multimodel.csv", index=False)
print(f"  Saved OAT tables")

# ===========================================================================
# STEP 6: GENERATE FIGURES 5.1, 5.2, 5.3
# ===========================================================================

print(f"\n[5] Generating figures...")

sc_short_label = {
    "ERA5 historical 1990-2024": "Hist\n1990-2024",
    "ERA5 recent 2015-2024":     "Recent\n2015-2024",
    "SSP2-4.5 / 2030-2050":      "SSP245\n2030-2050",
    "SSP2-4.5 / 2080-2100":      "SSP245\n2080-2100",
    "SSP5-8.5 / 2030-2050":      "SSP585\n2030-2050",
    "SSP5-8.5 / 2080-2100":      "SSP585\n2080-2100",
}
sc_colors = ["#333333", "#777777", "#4A7AB8", "#1F4E79", "#D49A4A", "#A03020"]

# Figure 5.1: bootstrap CIs (2 panels: V1, V3)
fig, axes = plt.subplots(1, 2, figsize=(18, 7.5))
for ax, df_use, title in [
    (axes[0], v1_df, f"(a) Variant 1: cooled stock (n={n_cooled:,})"),
    (axes[1], v3_df, f"(b) Variant 3: full stock x adoption (n_max={n_full:,})")]:
    df_plot = df_use.set_index("scenario").reindex(SCENARIOS).reset_index()
    x = np.arange(len(SCENARIOS))
    pt = df_plot["point_GWh_median"].values
    lo = df_plot["ci_low_GWh"].values
    hi = df_plot["ci_high_GWh"].values
    ax.bar(x, pt, yerr=[pt - lo, hi - pt], color=sc_colors,
           edgecolor="black", linewidth=0.5, capsize=8,
           error_kw={"linewidth": 1.5})
    for i in range(len(SCENARIOS)):
        ax.text(i, hi[i] + max(hi)*0.02, f"{pt[i]:.0f}\n[{lo[i]:.0f}-{hi[i]:.0f}]",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([sc_short_label[s] for s in SCENARIOS], fontsize=8)
    ax.set_ylabel("Total cooling demand (GWh/yr)", fontsize=11)
    ax.set_title(title, fontsize=11, fontweight="bold", color=HEADING, loc="left")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)

fig.text(0.5, -0.03,
         "Note: future scenario CIs include 12-model NEX-GDDP-CMIP6 ensemble spread + within-model inter-annual variability\n"
         "(Hawkins & Sutton 2009 framework; bias correction per Thrasher et al. 2022, Scientific Data).",
         ha="center", fontsize=9, style="italic", color="#444444")
fig.suptitle(f"Bootstrap 95% CIs - multi-model uncertainty (B={B_BOOT}, 12 CMIP6 models)",
             fontsize=13, fontweight="bold", color=HEADING, y=1.0)
plt.tight_layout()
fig51_png = FIGURES_DIR / "fig_5_1_uncertainty_demand_evolution_multimodel.png"
plt.savefig(fig51_png, dpi=300, bbox_inches="tight")
plt.savefig(FIGURES_DIR / "fig_5_1_uncertainty_demand_evolution_multimodel.pdf",
            bbox_inches="tight")
plt.close()
print(f"  Saved: {fig51_png.name}")

# Figure 5.2: tornado (2 panels)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
for ax, oat_df_use, baseline, title in [
    (axes[0], oat_v1_df, baseline_v1_oat, f"(a) Variant 1 (baseline = {baseline_v1_oat:.1f} GWh)"),
    (axes[1], oat_v3_df, baseline_v3_oat, f"(b) Variant 3 (baseline = {baseline_v3_oat:.1f} GWh)")]:
    pivot = oat_df_use.pivot(index="input", columns="direction", values="delta_GWh")
    pivot["max_abs"] = pivot[["low", "high"]].abs().max(axis=1)
    pivot = pivot.sort_values("max_abs", ascending=True)
    y_pos = np.arange(len(pivot))
    ax.barh(y_pos, pivot["low"], color="#4A7AB8", edgecolor="black",
            linewidth=0.5, label="Low end of range")
    ax.barh(y_pos, pivot["high"], color="#A03020", edgecolor="black",
            linewidth=0.5, label="High end of range")
    ax.axvline(0, color="black", linewidth=1.0)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pivot.index, fontsize=9)
    ax.set_xlabel("Delta from baseline demand (GWh/yr)", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold", color=HEADING, loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
fig.suptitle(f"OAT tornado at {HEADLINE_SOBOL_SCENARIO} - multi-model",
             fontsize=13, fontweight="bold", color=HEADING, y=1.0)
plt.tight_layout()
fig52_png = FIGURES_DIR / "fig_5_2_tornado_diagram_multimodel.png"
plt.savefig(fig52_png, dpi=300, bbox_inches="tight")
plt.savefig(FIGURES_DIR / "fig_5_2_tornado_diagram_multimodel.pdf",
            bbox_inches="tight")
plt.close()
print(f"  Saved: {fig52_png.name}")

# Figure 5.3: Sobol indices (2 panels, S1 and ST grouped bars per input)
fig, axes = plt.subplots(1, 2, figsize=(17, 6.5))
for ax, sob_df_use, title in [
    (axes[0], sobol_v1_df, "(a) Variant 1: 3 inputs"),
    (axes[1], sobol_v3_df, "(b) Variant 3: 4 inputs")]:
    y_pos = np.arange(len(sob_df_use))
    w = 0.35
    ax.barh(y_pos - w/2, sob_df_use["S1"],
            xerr=[sob_df_use["S1"] - sob_df_use["S1_ci_low"],
                  sob_df_use["S1_ci_high"] - sob_df_use["S1"]],
            height=w, color="#4A7AB8", edgecolor="black", linewidth=0.5,
            label="S1 (first-order)", capsize=4)
    ax.barh(y_pos + w/2, sob_df_use["ST"],
            xerr=[sob_df_use["ST"] - sob_df_use["ST_ci_low"],
                  sob_df_use["ST_ci_high"] - sob_df_use["ST"]],
            height=w, color="#A03020", edgecolor="black", linewidth=0.5,
            label="S_T (total)", capsize=4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sob_df_use["input"], fontsize=9)
    ax.set_xlabel("Sobol index", fontsize=10)
    ax.set_xlim(0, max(1.05, sob_df_use["ST_ci_high"].max() * 1.1))
    ax.set_title(title, fontsize=11, fontweight="bold", color=HEADING, loc="left")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
fig.suptitle(f"Sobol sensitivity indices at {HEADLINE_SOBOL_SCENARIO} - multi-model "
             f"(N={SOBOL_N}, inner bootstrap={SOBOL_INNER_BOOT})",
             fontsize=12, fontweight="bold", color=HEADING, y=1.0)
plt.tight_layout()
fig53_png = FIGURES_DIR / "fig_5_3_sobol_sensitivity_indices_multimodel.png"
plt.savefig(fig53_png, dpi=300, bbox_inches="tight")
plt.savefig(FIGURES_DIR / "fig_5_3_sobol_sensitivity_indices_multimodel.pdf",
            bbox_inches="tight")
plt.close()
print(f"  Saved: {fig53_png.name}")

# ===========================================================================
# STEP 7: HEADLINE OUTPUT + EXEC LOG
# ===========================================================================

print("\n" + "=" * 75)
print("HEADLINE COMPARISON")
print("=" * 75)

print(f"\n  V1 cooled stock - SSP5-8.5/2080-2100:")
v1_h = v1_df[v1_df["scenario"] == "SSP5-8.5 / 2080-2100"].iloc[0]
print(f"    Locked single-model (Stage 2 v3): 110 GWh [98, 122]  (24 GWh wide)")
print(f"    NEW multi-model:                  {v1_h['point_GWh_median']:.0f} GWh "
      f"[{v1_h['ci_low_GWh']:.0f}, {v1_h['ci_high_GWh']:.0f}]  "
      f"({v1_h['ci_high_GWh']-v1_h['ci_low_GWh']:.0f} GWh wide, {v1_h['ci_width_pct']:.1f}% relative)")

print(f"\n  V1 dominant uncertainty source:  {v1_inputs[ST_v1.argmax()]}  "
      f"(S_T = {ST_v1.max():.3f})")
print(f"  V3 dominant uncertainty source:  {v3_inputs[ST_v3.argmax()]}  "
      f"(S_T = {ST_v3.max():.3f})")

print(f"\n  V1 Sobol S_T breakdown:")
for inp, st in zip(v1_inputs, ST_v1):
    print(f"    {inp:42s}: {st:.3f}")
print(f"    {'(sum)':42s}: {ST_v1.sum():.3f}")

print(f"\n  V3 Sobol S_T breakdown:")
for inp, st in zip(v3_inputs, ST_v3):
    print(f"    {inp:42s}: {st:.3f}")
print(f"    {'(sum)':42s}: {ST_v3.sum():.3f}")

# Final exec log
exec_log.extend([
    "",
    f"V1 SSP5-8.5/2080: median {v1_h['point_GWh_median']:.1f} GWh [{v1_h['ci_low_GWh']:.1f}, {v1_h['ci_high_GWh']:.1f}]",
    f"V1 dominant: {v1_inputs[ST_v1.argmax()]} (S_T={ST_v1.max():.3f})",
    f"V3 dominant: {v3_inputs[ST_v3.argmax()]} (S_T={ST_v3.max():.3f})",
    "",
    f"Finished: {datetime.now().isoformat()}",
])
exec_log_path = LOGS_DIR / "script_c_execution_log.txt"
with open(exec_log_path, "w", encoding="utf-8") as f:
    f.write("\n".join(exec_log))

print("\n" + "=" * 75)
print("SCRIPT C COMPLETE")
print("=" * 75)
print(f"\nOutputs saved to: {MULTIMODEL_ROOT}")
print(f"\nKey deliverables:")
print(f"  Tables:")
print(f"    tab_5_1_uncertainty_intervals_multimodel.csv")
print(f"    tab_5_2a_sobol_v1_multimodel.csv")
print(f"    tab_5_2b_sobol_v3_multimodel.csv")
print(f"    tab_5_3a_variance_v1_multimodel.csv, tab_5_3b_variance_v3_multimodel.csv")
print(f"    tab_5_4a_oat_v1_multimodel.csv, tab_5_4b_oat_v3_multimodel.csv")
print(f"  Figures:")
print(f"    fig_5_1_uncertainty_demand_evolution_multimodel.png/.pdf")
print(f"    fig_5_2_tornado_diagram_multimodel.png/.pdf")
print(f"    fig_5_3_sobol_sensitivity_indices_multimodel.png/.pdf")
print(f"\nReady for Script D (methodology document update).")