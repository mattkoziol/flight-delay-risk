"""Build the app's data and model files from the assignment 5 and 6 outputs. Run once, locally, from assignment8/."""
import json
import os
import shutil

import numpy as np
import pandas as pd

from core import FEATURES, NUMS

A5 = os.path.join("..", "assignment5", "data", "clean")
A6 = os.path.join("..", "assignment6")
DISPLAY = ["Reporting_Airline", "carrier_name", "Flight_Number_Reporting_Airline", "Origin", "OriginCityName", "Dest",
           "DestCityName", "CRSDepTime", "CRSArrTime", "ArrDelay"]
EXACT = {"route_pad", "te_carrier", "te_origin", "te_dest", "te_route", "te_carrier_origin", "te_origin_hour"}

os.makedirs("data", exist_ok=True)
os.makedirs("models", exist_ok=True)

m = pd.read_parquet(os.path.join(A5, "model.parquet"), columns=["row_id", "FlightDate", "split", "delayed"] + FEATURES)
f = pd.read_parquet(os.path.join(A5, "flights.parquet"), columns=["row_id"] + DISPLAY)
p = pd.read_parquet(os.path.join(A6, "data", "preds.parquet"), columns=["row_id", "p_hgb"])
df = m.merge(f, on="row_id", how="left", validate="1:1").merge(p, on="row_id", how="left", validate="1:1")
assert df["p_hgb"].notna().all() and df["Origin"].notna().all()

for c in NUMS:
    if c not in EXACT:
        df[c] = df[c].astype("float32")
for c in ("Reporting_Airline", "carrier_name", "Origin", "OriginCityName", "Dest", "DestCityName"):
    df[c] = df[c].astype("category")
df["Flight_Number_Reporting_Airline"] = df["Flight_Number_Reporting_Airline"].astype("int32")
for c in ("CRSDepTime", "CRSArrTime"):
    df[c] = pd.to_numeric(df[c]).astype("int16")
df["ArrDelay"] = df["ArrDelay"].astype("float32")
df["delayed"] = df["delayed"].astype("int8")
df["split"] = df["split"].astype("category")
df = df.sort_values(["FlightDate", "Reporting_Airline", "Origin", "Dest", "CRSDepTime"], kind="mergesort")
df.to_parquet(os.path.join("data", "flights.parquet"), index=False, compression="zstd")

agg = (df.assign(month=df["FlightDate"].dt.month)
         .groupby(["month", "Reporting_Airline", "Origin", "weekday", "dep_hour"], observed=True)["delayed"]
         .agg(n="size", n_delayed="sum").reset_index())
agg = agg[agg["n"] > 0]
carriers = df.drop_duplicates("Reporting_Airline").set_index("Reporting_Airline")["carrier_name"].astype(str)
agg["carrier_name"] = agg["Reporting_Airline"].map(carriers)
agg.to_parquet(os.path.join("data", "agg.parquet"), index=False)

panel = pd.read_csv(os.path.join(A6, "output", "02_panel.csv")).set_index(["key", "op"])
t50 = panel.loc[("hgb", "t50")]
test = df[df["split"] == "test"]
metrics = dict(auc=t50["auc"], pr_auc=t50["pr_auc"], brier=t50["brier"], ece=t50["ece"], logloss=t50["logloss"],
               accuracy=t50["acc"], precision=t50["prec"], recall=t50["rec"], f1=t50["f1"],
               lr_auc=panel.loc[("lr", "t50"), "auc"], rf_auc=panel.loc[("rf", "t50"), "auc"],
               base_rate=float(df["delayed"].mean()), test_base_rate=float(test["delayed"].mean()),
               rows={s: int((df["split"] == s).sum()) for s in ("train", "valid", "test")},
               dates={s: [str(d.date()) for d in (g["FlightDate"].min(), g["FlightDate"].max())]
                      for s, g in df.groupby("split", observed=True)})
json.dump({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in metrics.items()},
          open(os.path.join("data", "metrics.json"), "w"), indent=2)
shutil.copy(os.path.join(A6, "models", "hgb.joblib"), os.path.join("models", "hgb.joblib"))

for d in ("data", "models"):
    for name in sorted(os.listdir(d)):
        print(f"{d}/{name:20s} {os.path.getsize(os.path.join(d, name)) / 1e6:7.1f} MB")
print(f"{len(df):,} flights, memory {df.memory_usage(deep=True).sum() / 1e6:.0f} MB")
