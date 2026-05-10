"""
============================================================================
SCRIPT A: Validate NEX-GDDP-CMIP6 CSVs and Build Multi-Model CDD Table
----------------------------------------------------------------------------
This single script does everything for the CDD revision step:

  1. Verifies all 60 input CSVs (existence, row counts, ranges, NaN check)
  2. Computes JJA daily CDD per model per scenario per year (base 22 C)
  3. Aggregates to per-period mean CDD per model
  4. Builds the ensemble multi-model CDD table
  5. Sanity-checks NEX-GDDP historical ensemble against ERA5 baseline
  6. Saves all outputs into the new _MULTIMODEL folder
  7. Generates a README and verification report

Inputs:
    DATA_FOLDER (60 CSV files from extraction step)
    ERA5 anchors (constants below, from existing locked Stage 2)

Outputs (in OUTPUT_FOLDER):
    _TABLES/tab_4_1_cdd_projections_multimodel.csv
    _TABLES/tab_4_1_cdd_per_model_per_year.csv
    _TABLES/verification_summary.csv
    _LOGS/verification_report.txt
    _LOGS/script_a_execution_log.txt
    _README.md

Methodology locks:
    - Cooling base temperature: 22 C (UNI 11300, unchanged from Session 3)
    - Season: JJA (June-July-August, unchanged)
    - Bias correction: NONE (NEX-GDDP is already bias-corrected via BCSD/SDM)
    - Anchor for relative changes: each model's own historical 1990-2014 mean

Runtime: 1-2 minutes
Idempotent: re-running overwrites previous outputs cleanly
============================================================================
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

# ===========================================================================
# CONFIGURATION
# ===========================================================================

# Input: where the 60 NEX-GDDP CSVs live
DATA_FOLDER = Path(
    r"C:\Users\n.mohammadi\Desktop\NimaMohammadi"
    r"\03. Nima Mohammadi - Thesis"
    r"\Data Collections\Future (CMIP6) Climate Datasets\DATA"
)

# Output: new folder inside ML Dataset thesis project
OUTPUT_FOLDER = Path(
    r"C:\Users\n.mohammadi\Desktop\NimaMohammadi"
    r"\02.Nima Mohammadi - Thesis\ML Dataset"
    r"\07_results\stage2\_MULTIMODEL"
)

# 12 models (must match extraction script exactly)
MODELS = [
    "MPI-ESM1-2-HR", "CMCC-ESM2", "EC-Earth3", "CNRM-ESM2-1",
    "IPSL-CM6A-LR", "CESM2", "HadGEM3-GC31-LL", "CanESM5",
    "NorESM2-MM", "GFDL-ESM4", "MIROC6", "ACCESS-ESM1-5",
]

# Scenario-period combinations (must match extraction)
EXTRACTIONS = [
    ("historical", 1990, 2014),
    ("ssp245",     2030, 2050),
    ("ssp245",     2080, 2100),
    ("ssp585",     2030, 2050),
    ("ssp585",     2080, 2100),
]

# Methodology constants (LOCKED from existing Stage 2)
COOLING_BASE_C = 22.0          # UNI 11300 cooling base
JJA_MONTHS = [6, 7, 8]         # June, July, August
KELVIN_TO_C = 273.15           # NEX-GDDP tas is in Kelvin

# ERA5 anchor values from your existing locked Stage 2 (for sanity check only)
ERA5_HISTORICAL_1990_2024 = 162.1   # CDD, your locked value
ERA5_RECENT_2015_2024 = 206.6       # CDD, your locked anchor

# Expected row counts (sanity check)
EXPECTED_DAYS_HISTORICAL = 9131    # 25 years 1990-2014 (6 leap years)
EXPECTED_DAYS_FUTURE = 7670        # 21 years (5 leap years)

# ===========================================================================
# SETUP: Create output folder structure
# ===========================================================================

print("=" * 75)
print("SCRIPT A: Validate & Build Multi-Model CDD Table")
print(f"Run started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 75)

# Create folder structure
TABLES_DIR = OUTPUT_FOLDER / "_TABLES"
FIGURES_DIR = OUTPUT_FOLDER / "_FIGURES"
LOGS_DIR = OUTPUT_FOLDER / "_LOGS"

for folder in [OUTPUT_FOLDER, TABLES_DIR, FIGURES_DIR, LOGS_DIR]:
    folder.mkdir(parents=True, exist_ok=True)

print(f"\nOutput folder: {OUTPUT_FOLDER}")
print(f"  _TABLES  : created/exists")
print(f"  _FIGURES : created/exists")
print(f"  _LOGS    : created/exists")

# Verify input folder exists
if not DATA_FOLDER.exists():
    print(f"\nERROR: Input folder does not exist: {DATA_FOLDER}")
    sys.exit(1)

# Open execution log
exec_log_path = LOGS_DIR / "script_a_execution_log.txt"
exec_log_lines = [
    f"Script A execution log",
    f"Started: {datetime.now().isoformat()}",
    f"Input folder: {DATA_FOLDER}",
    f"Output folder: {OUTPUT_FOLDER}",
    "=" * 75,
    "",
]

# ===========================================================================
# STEP 1: VERIFICATION OF ALL 60 CSV FILES
# ===========================================================================

print("\n" + "=" * 75)
print("STEP 1: VERIFY 60 INPUT CSV FILES")
print("=" * 75)

verification_rows = []
verification_pass = True
loaded_data = {}  # (model, scenario, period) -> DataFrame

for model in MODELS:
    for scenario, y_start, y_end in EXTRACTIONS:
        filename = f"CMIP6_Milan_{model}_{scenario}_{y_start}_{y_end}.csv"
        filepath = DATA_FOLDER / filename

        # Check file existence
        if not filepath.exists():
            verification_rows.append({
                "model": model, "scenario": scenario,
                "period": f"{y_start}-{y_end}", "filename": filename,
                "status": "MISSING_FILE", "n_rows": 0,
                "tas_min_C": np.nan, "tas_max_C": np.nan,
                "tas_mean_C": np.nan, "n_nan": 0,
                "issue": "File does not exist",
            })
            verification_pass = False
            print(f"  [FAIL] {filename}: MISSING")
            continue

        # Load CSV
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            verification_rows.append({
                "model": model, "scenario": scenario,
                "period": f"{y_start}-{y_end}", "filename": filename,
                "status": "READ_ERROR", "n_rows": 0,
                "tas_min_C": np.nan, "tas_max_C": np.nan,
                "tas_mean_C": np.nan, "n_nan": 0,
                "issue": str(e)[:200],
            })
            verification_pass = False
            print(f"  [FAIL] {filename}: read error - {e}")
            continue

        # Check required columns
        if "time" not in df.columns or "tas" not in df.columns:
            verification_rows.append({
                "model": model, "scenario": scenario,
                "period": f"{y_start}-{y_end}", "filename": filename,
                "status": "MISSING_COLUMNS", "n_rows": len(df),
                "tas_min_C": np.nan, "tas_max_C": np.nan,
                "tas_mean_C": np.nan, "n_nan": 0,
                "issue": f"Columns: {list(df.columns)}",
            })
            verification_pass = False
            print(f"  [FAIL] {filename}: missing time or tas column")
            continue

        # Convert tas to Celsius and compute stats
        df["tas_C"] = df["tas"] - KELVIN_TO_C
        n_rows = len(df)
        tas_min = df["tas_C"].min()
        tas_max = df["tas_C"].max()
        tas_mean = df["tas_C"].mean()
        n_nan = df["tas"].isna().sum()

        # Expected row count
        expected = EXPECTED_DAYS_HISTORICAL if scenario == "historical" else EXPECTED_DAYS_FUTURE

        # Sanity checks
        issues = []
        status = "PASS"

        if n_rows != expected:
            issues.append(f"row count {n_rows} (expected {expected})")
            status = "WARN"

        if n_nan > 0:
            issues.append(f"{n_nan} NaN values")
            status = "FAIL"
            verification_pass = False

        if tas_min < -40 or tas_max > 55:
            issues.append(f"value range out of bounds [{tas_min:.1f}, {tas_max:.1f}]")
            status = "FAIL"
            verification_pass = False

        if tas_mean < -10 or tas_mean > 30:
            issues.append(f"mean temperature suspect: {tas_mean:.1f} C")
            status = "WARN"

        # Parse dates
        try:
            df["date"] = pd.to_datetime(df["time"])
            df["year"] = df["date"].dt.year
            df["month"] = df["date"].dt.month
        except Exception as e:
            issues.append(f"date parse error: {e}")
            status = "FAIL"
            verification_pass = False

        verification_rows.append({
            "model": model, "scenario": scenario,
            "period": f"{y_start}-{y_end}", "filename": filename,
            "status": status, "n_rows": n_rows,
            "tas_min_C": round(tas_min, 2), "tas_max_C": round(tas_max, 2),
            "tas_mean_C": round(tas_mean, 2), "n_nan": int(n_nan),
            "issue": "; ".join(issues) if issues else "",
        })

        # Cache the DataFrame for Step 2
        loaded_data[(model, scenario, y_start, y_end)] = df

        marker = "[PASS]" if status == "PASS" else f"[{status}]"
        msg = f"  {marker} {filename}: {n_rows:,} rows, mean tas {tas_mean:.2f} C"
        if issues:
            msg += f" -- {'; '.join(issues)}"
        print(msg)

# Save verification summary
verification_df = pd.DataFrame(verification_rows)
verification_path = TABLES_DIR / "verification_summary.csv"
verification_df.to_csv(verification_path, index=False)

n_pass = (verification_df["status"] == "PASS").sum()
n_warn = (verification_df["status"] == "WARN").sum()
n_fail = (verification_df["status"] == "FAIL").sum()

print(f"\nVerification summary:")
print(f"  PASS: {n_pass} / {len(verification_df)}")
print(f"  WARN: {n_warn}")
print(f"  FAIL: {n_fail}")
print(f"  Saved: {verification_path}")

exec_log_lines.extend([
    "STEP 1 VERIFICATION:",
    f"  Total files checked: {len(verification_df)}",
    f"  Pass: {n_pass}, Warn: {n_warn}, Fail: {n_fail}",
    f"  Saved: verification_summary.csv",
    "",
])

if not verification_pass:
    print("\nERROR: verification failed for one or more files. See verification_summary.csv.")
    print("Cannot proceed to CDD computation. Fix issues and re-run.")
    sys.exit(1)

# ===========================================================================
# STEP 2: COMPUTE JJA CDD PER MODEL PER SCENARIO PER YEAR
# ===========================================================================

print("\n" + "=" * 75)
print("STEP 2: COMPUTE JJA CDD PER MODEL PER YEAR")
print("=" * 75)
print(f"  Cooling base: {COOLING_BASE_C} C (UNI 11300)")
print(f"  Season: JJA (months {JJA_MONTHS})")

per_year_rows = []

for (model, scenario, y_start, y_end), df in loaded_data.items():
    # Filter JJA only
    jja = df[df["month"].isin(JJA_MONTHS)].copy()

    # Daily CDD = max(0, tas_C - 22)
    jja["cdd_daily"] = (jja["tas_C"] - COOLING_BASE_C).clip(lower=0)

    # Sum to annual CDD
    annual_cdd = jja.groupby("year")["cdd_daily"].sum().reset_index()
    annual_cdd.columns = ["year", "cdd"]

    # Add identifiers
    annual_cdd["model"] = model
    annual_cdd["scenario"] = scenario
    annual_cdd["period_start"] = y_start
    annual_cdd["period_end"] = y_end

    per_year_rows.append(annual_cdd)

per_year_df = pd.concat(per_year_rows, ignore_index=True)
per_year_df = per_year_df[["model", "scenario", "period_start", "period_end", "year", "cdd"]]
per_year_df = per_year_df.sort_values(["model", "scenario", "period_start", "year"])

# Save per-year detail
per_year_path = TABLES_DIR / "tab_4_1_cdd_per_model_per_year.csv"
per_year_df.to_csv(per_year_path, index=False)
print(f"  Per-year CDD records: {len(per_year_df):,}")
print(f"  Saved: {per_year_path}")

exec_log_lines.extend([
    "STEP 2 CDD COMPUTATION:",
    f"  Total per-year records: {len(per_year_df):,}",
    f"  Saved: tab_4_1_cdd_per_model_per_year.csv",
    "",
])

# ===========================================================================
# STEP 3: AGGREGATE TO PER-PERIOD MEAN CDD PER MODEL
# ===========================================================================

print("\n" + "=" * 75)
print("STEP 3: PER-MODEL PER-PERIOD MEAN CDD")
print("=" * 75)

per_period = per_year_df.groupby(
    ["model", "scenario", "period_start", "period_end"]
).agg(
    cdd_mean=("cdd", "mean"),
    cdd_std=("cdd", "std"),
    cdd_min=("cdd", "min"),
    cdd_max=("cdd", "max"),
    n_years=("cdd", "count"),
).reset_index()

per_period["cdd_mean"] = per_period["cdd_mean"].round(1)
per_period["cdd_std"] = per_period["cdd_std"].round(1)
per_period["cdd_min"] = per_period["cdd_min"].round(1)
per_period["cdd_max"] = per_period["cdd_max"].round(1)

print(f"  Per-model per-period rows: {len(per_period)} (12 models x 5 periods)")

# ===========================================================================
# STEP 4: ENSEMBLE MULTI-MODEL CDD TABLE
# ===========================================================================

print("\n" + "=" * 75)
print("STEP 4: ENSEMBLE STATISTICS ACROSS 12 MODELS")
print("=" * 75)

ensemble_rows = []
for scenario, y_start, y_end in EXTRACTIONS:
    subset = per_period[
        (per_period["scenario"] == scenario) &
        (per_period["period_start"] == y_start) &
        (per_period["period_end"] == y_end)
    ]
    cdd_vals = subset["cdd_mean"].values

    ensemble_rows.append({
        "scenario": scenario,
        "period": f"{y_start}-{y_end}",
        "period_start": y_start,
        "period_end": y_end,
        "n_models": len(cdd_vals),
        "cdd_ensemble_median": round(np.median(cdd_vals), 1),
        "cdd_ensemble_mean": round(np.mean(cdd_vals), 1),
        "cdd_ensemble_p05": round(np.percentile(cdd_vals, 5), 1),
        "cdd_ensemble_p25": round(np.percentile(cdd_vals, 25), 1),
        "cdd_ensemble_p75": round(np.percentile(cdd_vals, 75), 1),
        "cdd_ensemble_p95": round(np.percentile(cdd_vals, 95), 1),
        "cdd_ensemble_min": round(np.min(cdd_vals), 1),
        "cdd_ensemble_max": round(np.max(cdd_vals), 1),
        "cdd_ensemble_std": round(np.std(cdd_vals), 1),
    })

ensemble_df = pd.DataFrame(ensemble_rows)

# Combined locked output: ensemble stats + per-model values for transparency
combined = ensemble_df.copy()
for model in MODELS:
    model_col = []
    for _, row in ensemble_df.iterrows():
        match = per_period[
            (per_period["model"] == model) &
            (per_period["scenario"] == row["scenario"]) &
            (per_period["period_start"] == row["period_start"])
        ]
        model_col.append(match["cdd_mean"].iloc[0] if len(match) > 0 else np.nan)
    combined[f"cdd_{model}"] = model_col

# Save the locked multi-model table
multimodel_path = TABLES_DIR / "tab_4_1_cdd_projections_multimodel.csv"
combined.to_csv(multimodel_path, index=False)
print(f"  Saved: {multimodel_path}")

# Print headline ensemble values
print(f"\n  Ensemble headlines (median across 12 models):")
for _, row in ensemble_df.iterrows():
    print(f"    {row['scenario']:12s} {row['period']:11s}: "
          f"median = {row['cdd_ensemble_median']:6.1f} CDD  "
          f"[5-95% range: {row['cdd_ensemble_p05']:5.1f} - {row['cdd_ensemble_p95']:5.1f}]  "
          f"({row['n_models']} models)")

exec_log_lines.extend([
    "STEP 4 ENSEMBLE:",
    f"  Ensemble headline values:",
])
for _, row in ensemble_df.iterrows():
    exec_log_lines.append(
        f"    {row['scenario']} {row['period']}: median {row['cdd_ensemble_median']} CDD "
        f"[p5-p95: {row['cdd_ensemble_p05']} - {row['cdd_ensemble_p95']}]"
    )
exec_log_lines.append("")

# ===========================================================================
# STEP 5: SANITY CHECK AGAINST ERA5
# ===========================================================================

print("\n" + "=" * 75)
print("STEP 5: SANITY CHECK - NEX-GDDP HISTORICAL vs ERA5")
print("=" * 75)

historical_row = ensemble_df[ensemble_df["scenario"] == "historical"].iloc[0]
nexgddp_hist = historical_row["cdd_ensemble_median"]
nexgddp_hist_mean = historical_row["cdd_ensemble_mean"]

bias_vs_era5 = nexgddp_hist - ERA5_HISTORICAL_1990_2024
pct_diff = 100.0 * bias_vs_era5 / ERA5_HISTORICAL_1990_2024

print(f"\n  ERA5 historical 1990-2024 (locked baseline):     {ERA5_HISTORICAL_1990_2024:.1f} CDD")
print(f"  NEX-GDDP ensemble historical 1990-2014 (median): {nexgddp_hist:.1f} CDD")
print(f"  NEX-GDDP ensemble historical 1990-2014 (mean):   {nexgddp_hist_mean:.1f} CDD")
print(f"  Difference (NEX-GDDP - ERA5):                    {bias_vs_era5:+.1f} CDD ({pct_diff:+.1f}%)")

# Interpretation
if abs(pct_diff) < 10:
    interp = "EXCELLENT agreement (< 10% difference)"
elif abs(pct_diff) < 20:
    interp = "GOOD agreement (10-20% difference)"
elif abs(pct_diff) < 30:
    interp = "ACCEPTABLE agreement (20-30%)"
else:
    interp = "WARNING: large discrepancy (> 30%)"

print(f"\n  Interpretation: {interp}")
print(f"\n  Note: ERA5 covers 1990-2024 (35 years).")
print(f"        NEX-GDDP historical covers 1990-2014 (25 years - model historical run).")
print(f"        Some difference is expected due to non-overlapping observation periods.")

exec_log_lines.extend([
    "STEP 5 ERA5 SANITY CHECK:",
    f"  ERA5 hist 1990-2024:     {ERA5_HISTORICAL_1990_2024:.1f} CDD",
    f"  NEX-GDDP hist median:    {nexgddp_hist:.1f} CDD",
    f"  Bias vs ERA5:            {bias_vs_era5:+.1f} CDD ({pct_diff:+.1f}%)",
    f"  Interpretation: {interp}",
    "",
])

# ===========================================================================
# STEP 6: WRITE VERIFICATION REPORT
# ===========================================================================

verif_report_path = LOGS_DIR / "verification_report.txt"
with open(verif_report_path, "w", encoding="utf-8") as f:
    f.write("=" * 75 + "\n")
    f.write("MULTI-MODEL CMIP6 CDD VERIFICATION REPORT\n")
    f.write("=" * 75 + "\n\n")
    f.write(f"Generated: {datetime.now().isoformat()}\n")
    f.write(f"Script: validate_and_build_cdd.py (Script A)\n\n")

    f.write("INPUT VERIFICATION\n")
    f.write("-" * 75 + "\n")
    f.write(f"Total files checked:      {len(verification_df)}\n")
    f.write(f"  PASS:                   {n_pass}\n")
    f.write(f"  WARN:                   {n_warn}\n")
    f.write(f"  FAIL:                   {n_fail}\n\n")

    f.write("METHODOLOGY\n")
    f.write("-" * 75 + "\n")
    f.write(f"Data source:              NEX-GDDP-CMIP6 (Thrasher et al. 2022)\n")
    f.write(f"Number of models:         12\n")
    f.write(f"Cooling base:             {COOLING_BASE_C} C (UNI 11300)\n")
    f.write(f"Season:                   JJA (June-July-August)\n")
    f.write(f"Bias correction:          NONE (NEX-GDDP is pre-bias-corrected via BCSD/SDM)\n\n")

    f.write("ENSEMBLE CDD VALUES (median across 12 models)\n")
    f.write("-" * 75 + "\n")
    for _, row in ensemble_df.iterrows():
        f.write(f"  {row['scenario']:12s} {row['period']:11s}: "
                f"median {row['cdd_ensemble_median']:6.1f}  "
                f"mean {row['cdd_ensemble_mean']:6.1f}  "
                f"[5-95%: {row['cdd_ensemble_p05']:5.1f} - {row['cdd_ensemble_p95']:5.1f}]  "
                f"std {row['cdd_ensemble_std']:.1f}\n")
    f.write("\n")

    f.write("ERA5 SANITY CHECK\n")
    f.write("-" * 75 + "\n")
    f.write(f"  ERA5 historical (locked):       {ERA5_HISTORICAL_1990_2024:.1f} CDD\n")
    f.write(f"  NEX-GDDP ensemble historical:   {nexgddp_hist:.1f} CDD (median)\n")
    f.write(f"  Bias:                           {bias_vs_era5:+.1f} CDD ({pct_diff:+.1f}%)\n")
    f.write(f"  Status:                         {interp}\n\n")

    f.write("FILES PRODUCED\n")
    f.write("-" * 75 + "\n")
    f.write("  _TABLES/tab_4_1_cdd_projections_multimodel.csv  (locked multi-model table)\n")
    f.write("  _TABLES/tab_4_1_cdd_per_model_per_year.csv       (full per-year detail)\n")
    f.write("  _TABLES/verification_summary.csv                  (per-file verification)\n")
    f.write("  _LOGS/verification_report.txt                     (this file)\n")
    f.write("  _LOGS/script_a_execution_log.txt                  (execution log)\n")

print(f"\n  Saved: {verif_report_path}")

# ===========================================================================
# STEP 7: WRITE README FOR _MULTIMODEL FOLDER
# ===========================================================================

readme_path = OUTPUT_FOLDER / "_README.md"
readme = f"""# Multi-Model Stage 2 Revision

This folder contains the multi-model CMIP6 revision of Stage 2 climate projections (Session 3 onwards). The single-model MPI-ESM1.2-HR + CIBSE delta-change methodology has been replaced with the NEX-GDDP-CMIP6 12-model ensemble.

## Status

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Stage: Script A complete (verification + CDD computation)

## Methodology summary

- Data source: NEX-GDDP-CMIP6 (Thrasher et al. 2022, Scientific Data)
- Models: 12 (MPI-ESM1-2-HR, CMCC-ESM2, EC-Earth3, CNRM-ESM2-1, IPSL-CM6A-LR, CESM2, HadGEM3-GC31-LL, CanESM5, NorESM2-MM, GFDL-ESM4, MIROC6, ACCESS-ESM1-5)
- Scenarios: SSP2-4.5 and SSP5-8.5
- Periods: historical 1990-2014, future 2030-2050 and 2080-2100
- Cooling base: 22 C (UNI 11300)
- Season: JJA (June-July-August)
- Bias correction: none additional (NEX-GDDP is already bias-corrected with BCSD/SDM)

## Folder structure

- `_TABLES/` - Main numerical outputs (CSV)
- `_FIGURES/` - Figures (filled by later scripts)
- `_LOGS/` - Verification reports and execution logs

## Headline ensemble CDD values (median across 12 models)

"""
for _, row in ensemble_df.iterrows():
    readme += (f"- {row['scenario']:12s} {row['period']:11s}: "
               f"{row['cdd_ensemble_median']:.1f} CDD "
               f"[5-95% range: {row['cdd_ensemble_p05']:.1f} - {row['cdd_ensemble_p95']:.1f}]\n")

readme += f"""
## ERA5 sanity check

NEX-GDDP ensemble historical median ({nexgddp_hist:.1f} CDD) vs ERA5 historical baseline ({ERA5_HISTORICAL_1990_2024:.1f} CDD): {bias_vs_era5:+.1f} CDD difference ({pct_diff:+.1f}%). {interp}.

## Next steps

- Script B: update Session 4 demand projections across 12 models
- Script C: update Session 5 uncertainty quantification with model-spread component
- Script D: update consolidated methodology document
"""

with open(readme_path, "w", encoding="utf-8") as f:
    f.write(readme)
print(f"  Saved: {readme_path}")

# ===========================================================================
# FINAL SUMMARY
# ===========================================================================

# Save execution log
exec_log_lines.extend([
    "=" * 75,
    f"Script A finished: {datetime.now().isoformat()}",
])
with open(exec_log_path, "w", encoding="utf-8") as f:
    f.write("\n".join(exec_log_lines))

print("\n" + "=" * 75)
print("SCRIPT A COMPLETE")
print("=" * 75)
print(f"\nAll outputs saved to:")
print(f"  {OUTPUT_FOLDER}")
print(f"\nKey deliverable for downstream scripts:")
print(f"  {multimodel_path}")
print(f"\nReady for Script B (multi-model demand projections).")