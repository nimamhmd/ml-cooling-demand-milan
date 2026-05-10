import pandas as pd

df = pd.read_csv(r"PATH_TO_buildings_with_epc_v2.csv")

print("Total rows:", len(df))
print(df["CLIMATIZZAZIONE_ESTIVA"].value_counts(dropna=False))
print(df["CLIMATIZZAZIONE_ESTIVA"].value_counts(normalize=True))