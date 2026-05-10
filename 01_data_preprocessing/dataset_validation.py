# dataset_validation.py
# Validate EPC→Building datasets (paths aligned with your screenshots)

from pathlib import Path
import pandas as pd

# ============================================================
# 1) CONFIG (EDIT ONLY THIS PART IF NEEDED)
# ============================================================

# Main CENED folder (from your screenshot)
BASE_CENED_DIR = Path(r"P:\Nima\23-11-2025\Data Collections\CENED")

# Folders you showed
SPATIAL_JOIN_DIR = BASE_CENED_DIR / "spatial_join_outputs"
BUILDING_LEVEL_DIR = BASE_CENED_DIR / "building_level_outputs"
VALIDATION_DIR = BASE_CENED_DIR / "validation_outputs"

# Expected files (we accept parquet OR csv)
EPC_BEST_STEM = "epc_best_per_building"
BUILDINGS_WITH_EPC_STEM = "buildings_with_epc"


# ============================================================
# 2) HELPERS
# ============================================================

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def find_dataset(folder: Path, stem: str) -> Path | None:
    """Find parquet first (preferred), otherwise csv."""
    pq = folder / f"{stem}.parquet"
    csv = folder / f"{stem}.csv"
    if pq.exists():
        return pq
    if csv.exists():
        return csv
    return None

def load_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    elif path.suffix.lower() == ".csv":
        # low_memory=False avoids dtype guessing chaos on large files
        return pd.read_csv(path, low_memory=False)
    else:
        raise ValueError(f"Unsupported file type: {path}")

def basic_report(df: pd.DataFrame, name: str) -> dict:
    rep = {}
    rep["dataset"] = name
    rep["rows"] = int(df.shape[0])
    rep["cols"] = int(df.shape[1])
    rep["columns"] = ", ".join(df.columns.astype(str).tolist())

    # duplicates (full-row)
    rep["duplicate_rows"] = int(df.duplicated().sum())

    # missingness overview (top 25 missing columns)
    miss = df.isna().mean().sort_values(ascending=False)
    top_miss = miss[miss > 0].head(25)
    rep["missing_cols_count"] = int((miss > 0).sum())
    rep["top_missing"] = "; ".join([f"{c}:{top_miss[c]:.3f}" for c in top_miss.index]) if len(top_miss) else "None"

    return rep

def guess_building_id_column(df: pd.DataFrame) -> str | None:
    """Try common building id column names."""
    candidates = [
        "building_id", "buildingid", "bld_id", "bldid",
        "EDIFC_ID", "EDIFC_ID ", "EDIFC_CASS", "FEATURE_ID",
        "id", "ID"
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in df.columns:
            return cand
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None

def save_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


# ============================================================
# 3) MAIN
# ============================================================

def main():
    print("[0] Checking folders...")
    print("BASE_CENED_DIR      :", BASE_CENED_DIR)
    print("SPATIAL_JOIN_DIR    :", SPATIAL_JOIN_DIR)
    print("BUILDING_LEVEL_DIR  :", BUILDING_LEVEL_DIR)
    print("VALIDATION_DIR      :", VALIDATION_DIR)

    if not BASE_CENED_DIR.exists():
        raise FileNotFoundError(f"BASE_CENED_DIR not found: {BASE_CENED_DIR}")
    if not SPATIAL_JOIN_DIR.exists():
        raise FileNotFoundError(f"SPATIAL_JOIN_DIR not found: {SPATIAL_JOIN_DIR}")
    if not BUILDING_LEVEL_DIR.exists():
        raise FileNotFoundError(f"BUILDING_LEVEL_DIR not found: {BUILDING_LEVEL_DIR}")

    ensure_dir(VALIDATION_DIR)

    print("\n[1] Locating input datasets...")

    epc_best_path = find_dataset(SPATIAL_JOIN_DIR, EPC_BEST_STEM)
    if epc_best_path is None:
        raise FileNotFoundError(
            f"Could not find {EPC_BEST_STEM}.(parquet/csv) in {SPATIAL_JOIN_DIR}"
        )

    bld_epc_path = find_dataset(BUILDING_LEVEL_DIR, BUILDINGS_WITH_EPC_STEM)
    if bld_epc_path is None:
        raise FileNotFoundError(
            f"Could not find {BUILDINGS_WITH_EPC_STEM}.(parquet/csv) in {BUILDING_LEVEL_DIR}"
        )

    print("✔ Found epc_best_per_building:", epc_best_path)
    print("✔ Found buildings_with_epc   :", bld_epc_path)

    print("\n[2] Loading data (this may take some time for CSV)...")
    epc_best = load_df(epc_best_path)
    bld_epc = load_df(bld_epc_path)

    print("   epc_best rows:", len(epc_best), "cols:", epc_best.shape[1])
    print("   bld_epc  rows:", len(bld_epc),  "cols:", bld_epc.shape[1])

    # ------------------------------------------------------------
    # 3A) Basic dataset reports
    # ------------------------------------------------------------
    print("\n[3] Building basic reports...")
    r1 = basic_report(epc_best, "epc_best_per_building")
    r2 = basic_report(bld_epc, "buildings_with_epc")

    report_lines = []
    for r in [r1, r2]:
        report_lines.append(f"== {r['dataset']} ==")
        report_lines.append(f"rows={r['rows']}  cols={r['cols']}")
        report_lines.append(f"duplicate_rows={r['duplicate_rows']}")
        report_lines.append(f"missing_cols_count={r['missing_cols_count']}")
        report_lines.append(f"top_missing={r['top_missing']}")
        report_lines.append("")

    # ------------------------------------------------------------
    # 3B) Key consistency checks (building_id)
    # ------------------------------------------------------------
    print("\n[4] Checking building ID consistency...")
    epc_id_col = guess_building_id_column(epc_best)
    bld_id_col = guess_building_id_column(bld_epc)

    report_lines.append("== Building ID detection ==")
    report_lines.append(f"epc_best building-id column: {epc_id_col}")
    report_lines.append(f"bld_epc  building-id column: {bld_id_col}")
    report_lines.append("")

    if epc_id_col is None or bld_id_col is None:
        report_lines.append("WARNING: Could not reliably detect building_id column in one/both datasets.")
        report_lines.append("Please check columns manually and update candidates list in guess_building_id_column().")
        report_lines.append("")
        print("⚠ Could not detect building_id column reliably.")
    else:
        # Normalize to string (IDs often behave badly if numeric)
        epc_ids = epc_best[epc_id_col].astype(str)
        bld_ids = bld_epc[bld_id_col].astype(str)

        epc_unique = set(epc_ids.dropna().unique())
        bld_unique = set(bld_ids.dropna().unique())

        inter = epc_unique.intersection(bld_unique)
        only_epc = epc_unique - bld_unique
        only_bld = bld_unique - epc_unique

        report_lines.append("== Building ID overlap ==")
        report_lines.append(f"unique IDs in epc_best: {len(epc_unique)}")
        report_lines.append(f"unique IDs in bld_epc : {len(bld_unique)}")
        report_lines.append(f"overlap (intersection): {len(inter)}")
        report_lines.append(f"IDs only in epc_best  : {len(only_epc)}")
        report_lines.append(f"IDs only in bld_epc   : {len(only_bld)}")
        report_lines.append("")

        # Small samples for debugging
        sample_only_epc = list(only_epc)[:20]
        sample_only_bld = list(only_bld)[:20]
        report_lines.append("sample IDs only in epc_best (first 20): " + (", ".join(sample_only_epc) if sample_only_epc else "None"))
        report_lines.append("sample IDs only in bld_epc  (first 20): " + (", ".join(sample_only_bld) if sample_only_bld else "None"))
        report_lines.append("")

        print("   Detected epc id col:", epc_id_col)
        print("   Detected bld id col:", bld_id_col)
        print("   Overlap:", len(inter))

    # ------------------------------------------------------------
    # 3C) Export validation outputs
    # ------------------------------------------------------------
    print("\n[5] Saving validation outputs...")
    out_txt = VALIDATION_DIR / "dataset_validation_report.txt"
    save_text(out_txt, "\n".join(report_lines))
    print("✔ Saved:", out_txt)

    # Save missingness table (useful for ML readiness)
    miss_bld = bld_epc.isna().mean().sort_values(ascending=False).reset_index()
    miss_bld.columns = ["column", "missing_ratio"]
    miss_path = VALIDATION_DIR / "buildings_with_epc_missingness.csv"
    miss_bld.to_csv(miss_path, index=False)
    print("✔ Saved:", miss_path)

    print("\n✅ DONE: Dataset validation completed.")
    print("Next step after validation: decide target variable(s) + feature set for ML and build the ML-ready table.")


if __name__ == "__main__":
    main()
