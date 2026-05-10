# ============================================================
# MILAN COOLING THESIS – REPRESENTATIVENESS TEST
# Purpose:
#   Verify whether the labeled subset (~19,064 with known cooling label)
#   is representative of the full building stock (53,041 DBT buildings).
# Input:
#   ml_dataset_buildings_epc_climate.csv
# Output:
#   representativeness_outputs_<timestamp>/
# ============================================================

import os
import json
from datetime import datetime

import numpy as np
import pandas as pd

# Optional stats tests
try:
    from scipy.stats import ks_2samp
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# Optional plotting
try:
    import matplotlib.pyplot as plt
    MPL_OK = True
except Exception:
    MPL_OK = False


# -----------------------------
# USER CONFIG
# -----------------------------
DATA_PATH = r"P:\Nima\23-11-2025\Data Collections\CENED\ml_dataset\ml_dataset_buildings_epc_climate.csv"

LABEL_COL = "CLIMATIZZAZIONE_ESTIVA"

CATEGORICAL_COLS = [
    "building_type_x",
    "building_use_x",
    "building_status_x",
    "CLASSE_x",
    "EDIFC_CASS_x",
]

NUMERIC_COLS = [
    "area_m2_x",
    "CDD26",
    "mean_temp_summer",
    "mean_temp",
    "max_temp",
    "RH_mean",
    "HI_mean",
    "ts_anom",
    "ts_anom_summer",
    "centroid_lat_x",
    "centroid_lon_x",
]

# For numeric hist plots (keep small to reduce clutter)
PLOT_NUMERIC = [
    "area_m2_x",
    "CDD26",
    "mean_temp_summer",
    "max_temp",
    "RH_mean",
    "HI_mean",
    "ts_anom_summer",
]

TOP_K_CATEGORIES = 25  # show top categories; rest grouped in "OTHER"


def now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_mkdir(p):
    os.makedirs(p, exist_ok=True)


def group_other(series, top_k=25):
    vc = series.value_counts(dropna=False)
    keep = set(vc.head(top_k).index.tolist())
    s2 = series.copy()
    s2 = s2.where(s2.isin(keep), other="OTHER")
    return s2


def compare_categorical(df_all, df_lab, col):
    # Group rare categories into OTHER for stable comparison
    a = group_other(df_all[col].astype("object"), TOP_K_CATEGORIES)
    b = group_other(df_lab[col].astype("object"), TOP_K_CATEGORIES)

    dist_all = a.value_counts(dropna=False, normalize=True)
    dist_lab = b.value_counts(dropna=False, normalize=True)

    # align index
    idx = sorted(set(dist_all.index).union(set(dist_lab.index)), key=lambda x: str(x))
    dist_all = dist_all.reindex(idx, fill_value=0.0)
    dist_lab = dist_lab.reindex(idx, fill_value=0.0)

    out = pd.DataFrame({
        "feature": col,
        "category": [str(i) for i in idx],
        "share_all": dist_all.values,
        "share_labeled": dist_lab.values,
        "abs_diff": np.abs(dist_all.values - dist_lab.values),
    })
    out["rel_diff"] = np.where(out["share_all"] > 0, out["abs_diff"] / out["share_all"], np.nan)
    return out


def compare_numeric(df_all, df_lab, col):
    a = pd.to_numeric(df_all[col], errors="coerce")
    b = pd.to_numeric(df_lab[col], errors="coerce")

    # summary stats
    def stats(s):
        s = s.dropna()
        if len(s) == 0:
            return {
                "n": 0, "mean": np.nan, "std": np.nan,
                "p05": np.nan, "p25": np.nan, "p50": np.nan, "p75": np.nan, "p95": np.nan
            }
        return {
            "n": int(len(s)),
            "mean": float(s.mean()),
            "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
            "p05": float(s.quantile(0.05)),
            "p25": float(s.quantile(0.25)),
            "p50": float(s.quantile(0.50)),
            "p75": float(s.quantile(0.75)),
            "p95": float(s.quantile(0.95)),
        }

    sa = stats(a)
    sb = stats(b)

    row = {
        "feature": col,
        "all_n": sa["n"],
        "lab_n": sb["n"],
        "all_mean": sa["mean"],
        "lab_mean": sb["mean"],
        "all_std": sa["std"],
        "lab_std": sb["std"],
        "all_p05": sa["p05"],
        "lab_p05": sb["p05"],
        "all_p50": sa["p50"],
        "lab_p50": sb["p50"],
        "all_p95": sa["p95"],
        "lab_p95": sb["p95"],
    }

    # KS test (if scipy available)
    if SCIPY_OK and sa["n"] > 50 and sb["n"] > 50:
        # Use finite values only
        a2 = a.dropna()
        b2 = b.dropna()
        ks_stat, ks_p = ks_2samp(a2, b2)
        row["ks_stat"] = float(ks_stat)
        row["ks_pvalue"] = float(ks_p)
    else:
        row["ks_stat"] = np.nan
        row["ks_pvalue"] = np.nan

    # Simple effect size proxy: standardized mean difference
    pooled_std = np.sqrt((row["all_std"]**2 + row["lab_std"]**2) / 2) if np.isfinite(row["all_std"]) and np.isfinite(row["lab_std"]) else np.nan
    row["std_mean_diff"] = (row["lab_mean"] - row["all_mean"]) / pooled_std if pooled_std and pooled_std > 0 else np.nan

    return row


def plot_numeric(df_all, df_lab, col, out_png):
    if not MPL_OK:
        return
    a = pd.to_numeric(df_all[col], errors="coerce").dropna()
    b = pd.to_numeric(df_lab[col], errors="coerce").dropna()
    if len(a) == 0 or len(b) == 0:
        return

    plt.figure()
    plt.hist(a, bins=40, alpha=0.5, label="All (53,041)")
    plt.hist(b, bins=40, alpha=0.5, label="Labeled (~19,064)")
    plt.title(f"Distribution comparison: {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def plot_categorical(comp_df, out_png, feature):
    if not MPL_OK:
        return

    # take top categories by share_all
    tmp = comp_df.copy()
    tmp = tmp.sort_values("share_all", ascending=False).head(15)

    x = np.arange(len(tmp))
    width = 0.45

    plt.figure(figsize=(10, 4))
    plt.bar(x - width/2, tmp["share_all"], width, label="All")
    plt.bar(x + width/2, tmp["share_labeled"], width, label="Labeled")
    plt.xticks(x, tmp["category"], rotation=45, ha="right")
    plt.title(f"Category shares: {feature} (top 15)")
    plt.ylabel("Share")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


def main():
    if not os.path.exists(DATA_PATH):
        print("❌ DATA_PATH not found:")
        print(DATA_PATH)
        return

    out_dir = os.path.join(os.getcwd(), f"representativeness_outputs_{now_stamp()}")
    plots_dir = os.path.join(out_dir, "plots")
    safe_mkdir(out_dir)
    safe_mkdir(plots_dir)

    print("🔄 Loading master ML dataset...")
    df = pd.read_csv(DATA_PATH, low_memory=False)
    print("✅ Loaded:", df.shape)

    if LABEL_COL not in df.columns:
        print(f"❌ Label column '{LABEL_COL}' not found.")
        return

    df_all = df.copy()
    df_lab = df[df[LABEL_COL].notna()].copy()

    # Summary counts
    summary = {
        "total_buildings": int(len(df_all)),
        "labeled_buildings": int(len(df_lab)),
        "labeled_share": float(len(df_lab) / max(len(df_all), 1)),
        "label_value_counts": df[LABEL_COL].value_counts(dropna=False).to_dict(),
        "scipy_available": SCIPY_OK,
        "matplotlib_available": MPL_OK,
    }

    with open(os.path.join(out_dir, "summary_counts.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n📌 Summary saved to summary_counts.json")
    print("Total:", summary["total_buildings"], "Labeled:", summary["labeled_buildings"])

    # Check columns exist
    cat_cols = [c for c in CATEGORICAL_COLS if c in df.columns]
    num_cols = [c for c in NUMERIC_COLS if c in df.columns]

    # Categorical comparisons
    cat_all = []
    for col in cat_cols:
        comp = compare_categorical(df_all, df_lab, col)
        cat_all.append(comp)
        plot_categorical(comp, os.path.join(plots_dir, f"cat_{col}.png"), col)

    if cat_all:
        cat_df = pd.concat(cat_all, ignore_index=True)
        cat_df.to_csv(os.path.join(out_dir, "categorical_distribution_comparison.csv"), index=False)
        print("✅ Saved categorical_distribution_comparison.csv")

    # Numeric comparisons + KS tests
    num_rows = []
    for col in num_cols:
        num_rows.append(compare_numeric(df_all, df_lab, col))
        if col in PLOT_NUMERIC:
            plot_numeric(df_all, df_lab, col, os.path.join(plots_dir, f"num_{col}.png"))

    num_df = pd.DataFrame(num_rows)
    num_df.to_csv(os.path.join(out_dir, "numeric_distribution_comparison.csv"), index=False)
    print("✅ Saved numeric_distribution_comparison.csv")

    # Extract KS-only view for quick decision
    ks_df = num_df[["feature", "ks_stat", "ks_pvalue", "std_mean_diff", "all_mean", "lab_mean", "all_p50", "lab_p50"]].copy()
    ks_df.to_csv(os.path.join(out_dir, "kolmogorov_smirnov_tests.csv"), index=False)
    print("✅ Saved kolmogorov_smirnov_tests.csv")

    print("\n✅ Representativeness check complete.")
    print("Output folder:", out_dir)
    print("\nDecision guidance:")
    print("- If most std_mean_diff are small (e.g., |SMD| < 0.2) and category shares are similar → proceed Stage 1.")
    print("- If strong shifts exist (e.g., |SMD| > 0.5 or big category abs_diff) → labeled set is biased; we should recover more matches or apply weighting.")


if __name__ == "__main__":
    main()