import pandas as pd
from pathlib import Path

# =============================
# 1) PATH CONFIGURATION
# =============================

DATA_PATH = Path(r"P:\Nima\23-11-2025\Data Collections\CENED\ml_dataset\ml_dataset_buildings_epc_climate.csv")

print("\n[1] Loading ML dataset...")
df = pd.read_csv(DATA_PATH)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# =============================
# 2) BASIC INFO
# =============================

print("\n[2] Data Types:")
print(df.dtypes.head(20))

# =============================
# 3) Missing Values
# =============================

print("\n[3] Missing values (top 20 columns):")
missing = df.isna().sum().sort_values(ascending=False)
print(missing.head(20))

# =============================
# 4) Numeric Columns Only
# =============================

numeric_df = df.select_dtypes(include=["number"])
print("\n[4] Numeric columns count:", numeric_df.shape[1])

# =============================
# 5) Check Cooling-Related Columns
# =============================

cooling_columns = [col for col in df.columns if "raffresc" in col.lower() or "estate" in col.lower()]
print("\n[5] Possible cooling-related columns:")
print(cooling_columns)

print("\n✅ Dataset inspection complete.")
