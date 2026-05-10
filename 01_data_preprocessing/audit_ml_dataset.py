# ============================================================
# MILAN COOLING THESIS – FULL DATASET AUDIT (ALL CSVs)
# Author: Nima Mohammadi
# Purpose:
#   1) Audit ALL datasets in a folder (features + quality checks)
#   2) Build a master feature inventory (avoid missing useful variables)
#   3) Flag potential leakage columns for Stage 1
# Outputs:
#   ./audit_outputs_<timestamp>/
# ============================================================

import os
import sys
import json
import time
from datetime import datetime

import numpy as np
import pandas as pd


# -----------------------------
# USER CONFIG
# -----------------------------
DATA_ROOT = r"P:\Nima\23-11-2025\Data Collections\CENED\ml_dataset"

# If your datasets are in subfolders under DATA_ROOT, keep True.
RECURSIVE_SEARCH = True

# Limit rows to speed up very large files (None = full file)
# Recommended: None for ML datasets (53k rows is fine)
READ_NROWS = None

# Potential targets in your project (script will analyze if present)
POSSIBLE_TARGETS = [
    "y_cooling_present",
    "CLIMATIZZAZIONE_ESTIVA",
    "y_cooled_area_m2",
    "SUPERF_UTILE_RAFFRESCATA",
]

# Leakage keywords (case-insensitive match on column names)
LEAKAGE_KEYWORDS = [
    "CE_", "RAFFRESC", "TELERAFF", "COOL", "CLIMAT", "CONDIZION",
    "SUPERF_UTILE_RAFFRESCATA", "VOLUME_LORDO_RAFFRESCATO"
]

# Columns that are IDs/metadata (not features). You can extend this list later.
ID_LIKE_HINTS = [
    "id", "ID", "cod", "COD", "pk", "PK", "uuid", "UUID",
    "geometry", "geom", "wkt", "shape", "fid", "Fid", "F_ID"
]


# -----------------------------
# HELPERS
# -----------------------------
def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def find_csv_files(root: str, recursive: bool = True) -> list[str]:
    csvs = []
    if recursive:
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(".csv"):
                    csvs.append(os.path.join(dirpath, fn))
    else:
        for fn in os.listdir(root):
            if fn.lower().endswith(".csv"):
                csvs.append(os.path.join(root, fn))
    return sorted(csvs)


def detect_leakage_columns(columns: list[str]) -> list[str]:
    leakage = []
    for col in columns:
        col_low = col.lower()
        for kw in LEAKAGE_KEYWORDS:
            if kw.lower() in col_low:
                leakage.append(col)
                break
    return sorted(set(leakage))


def guess_id_like_columns(columns: list[str]) -> list[str]:
    id_like = []
    for col in columns:
        col_low = col.lower()
        for hint in ID_LIKE_HINTS:
            if hint.lower() in col_low:
                id_like.append(col)
                break
    return sorted(set(id_like))


def dataset_basic_report(df: pd.DataFrame) -> dict:
    report = {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": df.columns.tolist(),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }
    return report


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    mr = df.isna().mean().sort_values(ascending=False)
    out = pd.DataFrame({"column": mr.index, "missing_ratio": mr.values})
    return out


def duplicate_rows_info(df: pd.DataFrame) -> dict:
    # Full-row duplicates (can be expensive but for 53k rows it is ok)
    dup_count = int(df.duplicated().sum())
    return {"duplicate_rows": dup_count, "duplicate_ratio": dup_count / max(len(df), 1)}


def analyze_targets(df: pd.DataFrame) -> dict:
    results = {}
    for t in POSSIBLE_TARGETS:
        if t in df.columns:
            s = df[t]
            # value counts with NaNs
            vc = s.value_counts(dropna=False)
            # convert to JSON-friendly
            results[t] = {str(k): int(v) for k, v in vc.items()}
    return results


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] == 0:
        return pd.DataFrame()
    return num.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).T


def top_correlations_with_target(df: pd.DataFrame, target_col: str, top_k: int = 25) -> pd.DataFrame:
    # Only numeric correlations
    if target_col not in df.columns:
        return pd.DataFrame()
    if not pd.api.types.is_numeric_dtype(df[target_col]):
        # If target is object/bool, try to map common cases
        mapped = df[target_col].copy()
        if mapped.dtype == "bool":
            mapped = mapped.astype(int)
        else:
            # Attempt mapping for typical labels
            # e.g., 'SI'/'NO', 1/0 strings, etc.
            mapping = {"SI": 1, "SÌ": 1, "YES": 1, "Y": 1, "TRUE": 1,
                       "NO": 0, "N": 0, "FALSE": 0}
            mapped = mapped.astype(str).str.strip().str.upper().map(mapping)
        if mapped.isna().all():
            return pd.DataFrame()
        temp = df.copy()
        temp[target_col] = mapped
        return top_correlations_with_target(temp, target_col, top_k=top_k)

    num = df.select_dtypes(include=[np.number]).copy()
    if target_col not in num.columns:
        return pd.DataFrame()

    corr = num.corr(numeric_only=True)[target_col].drop(labels=[target_col], errors="ignore")
    corr = corr.dropna().sort_values(key=lambda s: s.abs(), ascending=False).head(top_k)
    return pd.DataFrame({"feature": corr.index, "corr_with_target": corr.values})


# -----------------------------
# MAIN
# -----------------------------
def main():
    if not os.path.exists(DATA_ROOT):
        print(f"❌ DATA_ROOT does not exist:\n{DATA_ROOT}")
        sys.exit(1)

    ts = now_stamp()
    out_dir = os.path.join(os.getcwd(), f"audit_outputs_{ts}")
    safe_mkdir(out_dir)

    print("============================================================")
    print("FULL DATASET AUDIT – START")
    print("DATA_ROOT:", DATA_ROOT)
    print("RECURSIVE_SEARCH:", RECURSIVE_SEARCH)
    print("OUTPUT DIR:", out_dir)
    print("============================================================\n")

    csv_files = find_csv_files(DATA_ROOT, recursive=RECURSIVE_SEARCH)
    if not csv_files:
        print(f"❌ No CSV files found in:\n{DATA_ROOT}")
        sys.exit(1)

    print(f"✅ Found {len(csv_files)} CSV file(s).")
    for p in csv_files:
        print(" -", p)
    print("\n")

    # Master inventory across all datasets
    feature_inventory_rows = []
    dataset_index_rows = []

    for i, csv_path in enumerate(csv_files, start=1):
        base = os.path.basename(csv_path)
        name_no_ext = os.path.splitext(base)[0]
        safe_name = name_no_ext.replace(" ", "_").replace(".", "_")

        print(f"------------------------------------------------------------")
        print(f"[{i}/{len(csv_files)}] Auditing: {csv_path}")
        print(f"------------------------------------------------------------")

        # Load
        try:
            df = pd.read_csv(csv_path, nrows=READ_NROWS, low_memory=False)
        except UnicodeDecodeError:
            # fallback for some Italian CSV encodings
            df = pd.read_csv(csv_path, nrows=READ_NROWS, low_memory=False, encoding="latin1")

        # Basic report
        basic = dataset_basic_report(df)
        leak_cols = detect_leakage_columns(basic["columns"])
        id_like_cols = guess_id_like_columns(basic["columns"])
        miss = missing_report(df)
        dups = duplicate_rows_info(df)
        targets = analyze_targets(df)
        numsum = numeric_summary(df)

        # Save per-dataset artifacts
        per_dir = os.path.join(out_dir, safe_name)
        safe_mkdir(per_dir)

        with open(os.path.join(per_dir, "basic_report.json"), "w", encoding="utf-8") as f:
            json.dump({
                "file_path": csv_path,
                "file_name": base,
                "basic": basic,
                "duplicates": dups,
                "detected_leakage_columns": leak_cols,
                "guessed_id_like_columns": id_like_cols,
                "targets_found_value_counts": targets
            }, f, indent=2, ensure_ascii=False)

        miss.to_csv(os.path.join(per_dir, "missing_values_report.csv"), index=False)
        pd.DataFrame({"cooling_related_columns": leak_cols}).to_csv(
            os.path.join(per_dir, "cooling_related_columns.csv"), index=False
        )
        pd.DataFrame({"id_like_columns_guess": id_like_cols}).to_csv(
            os.path.join(per_dir, "id_like_columns_guess.csv"), index=False
        )

        if not numsum.empty:
            numsum.to_csv(os.path.join(per_dir, "numeric_summary.csv"))

        # Correlation checks for any targets present
        for tcol in targets.keys():
            corr_df = top_correlations_with_target(df, tcol, top_k=30)
            if not corr_df.empty:
                corr_df.to_csv(os.path.join(per_dir, f"top_corr_with_{tcol}.csv"), index=False)

        # Update dataset index
        dataset_index_rows.append({
            "file_name": base,
            "file_path": csv_path,
            "n_rows": basic["n_rows"],
            "n_cols": basic["n_cols"],
            "duplicate_rows": dups["duplicate_rows"],
            "n_leakage_cols_detected": len(leak_cols),
            "targets_found": ", ".join(list(targets.keys())) if targets else ""
        })

        # Update master inventory (one row per feature per dataset)
        for col in basic["columns"]:
            feature_inventory_rows.append({
                "file_name": base,
                "feature": col,
                "dtype": basic["dtypes"].get(col, ""),
                "missing_ratio": float(miss.loc[miss["column"] == col, "missing_ratio"].values[0]),
                "is_leakage_keyword_hit": col in leak_cols,
                "is_id_like_guess": col in id_like_cols,
                "is_target_candidate": col in POSSIBLE_TARGETS
            })

        print(f"✅ Saved reports to: {per_dir}\n")

    # Save master outputs
    dataset_index_df = pd.DataFrame(dataset_index_rows).sort_values(["file_name"])
    dataset_index_df.to_csv(os.path.join(out_dir, "DATASET_INDEX.csv"), index=False)

    feature_inventory_df = pd.DataFrame(feature_inventory_rows)
    feature_inventory_df.to_csv(os.path.join(out_dir, "MASTER_FEATURE_INVENTORY.csv"), index=False)

    # Create a compact “feature frequency” table across datasets
    freq = (feature_inventory_df.groupby("feature")
            .agg(
                datasets_count=("file_name", "nunique"),
                any_leakage=("is_leakage_keyword_hit", "max"),
                any_id_like=("is_id_like_guess", "max"),
                any_target_candidate=("is_target_candidate", "max"),
                dtypes_seen=("dtype", lambda s: ", ".join(sorted(set(map(str, s))))),
                avg_missing_ratio=("missing_ratio", "mean"),
            )
            .reset_index()
            .sort_values(["datasets_count", "any_leakage", "avg_missing_ratio"], ascending=[False, False, True]))

    freq.to_csv(os.path.join(out_dir, "FEATURE_FREQUENCY_ACROSS_DATASETS.csv"), index=False)

    print("============================================================")
    print("FULL DATASET AUDIT – COMPLETE")
    print("Output folder:")
    print(out_dir)
    print("\nKey outputs to open first:")
    print(" - DATASET_INDEX.csv")
    print(" - MASTER_FEATURE_INVENTORY.csv")
    print(" - FEATURE_FREQUENCY_ACROSS_DATASETS.csv")
    print("============================================================")


if __name__ == "__main__":
    main()