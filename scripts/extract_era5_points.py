"""Extract per-reservoir ERA5 daily features and write a git-sized CSV.

Reads data/raw/era5/era5_land_peninsular_india_*.nc (swvl1, sro, e) and
extracts daily values at each reservoir's (lat, lon) — exactly the point
extraction main.py::build_datasets performs — then writes
data/raw/era5/reservoir_era5_daily.csv with columns:

    Date, <res_id>_runoff, <res_id>_evap, <res_id>_soil_moisture   (x10 reservoirs)

Unit conventions mirror build_datasets: sro/e are meters -> x1000 (mm);
swvl1 is m3/m3. The CSV is committed to git so Kaggle needs no NetCDF upload.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ERA5_DIR = PROJECT_ROOT / "data" / "raw" / "era5"
OUT = ERA5_DIR / "reservoir_era5_daily.csv"
START, END = "2010-01-01", "2024-12-31"


def main() -> int:
    with open(PROJECT_ROOT / "configs" / "reservoirs.yaml") as f:
        reservoirs = yaml.safe_load(f)["reservoirs"]

    files = sorted(ERA5_DIR.glob("*.nc"))
    if not files:
        print(f"No NetCDF files in {ERA5_DIR}")
        return 1
    print(f"Opening {len(files)} ERA5 files...")
    ds = xr.open_mfdataset(files, combine="by_coords")

    out = None
    for res in reservoirs:
        res_id, lat, lon = res["id"], res["latitude"], res["longitude"]
        pt = ds.sel(latitude=lat, longitude=lon, method="nearest")
        pdf = pt.to_dataframe().select_dtypes(include=[np.number]).resample("D").mean()
        sub = pd.DataFrame(index=pdf.index)
        sub[f"{res_id}_runoff"] = pdf["sro"] * 1000.0 if "sro" in pdf else np.nan
        sub[f"{res_id}_evap"] = pdf["e"] * 1000.0 if "e" in pdf else np.nan
        sub[f"{res_id}_soil_moisture"] = pdf["swvl1"] if "swvl1" in pdf else np.nan
        out = sub if out is None else out.join(sub, how="outer")
        print(f"  {res_id}: extracted (nearest grid point used)")

    out.index.name = "Date"
    out = out.sort_index().loc[START:END]
    nan_total = int(out.isna().sum().sum())
    print(f"window {START}..{END}: {out.shape[0]} rows x {out.shape[1]} cols, NaNs={nan_total}")
    for c in [out.columns[0], out.columns[1], out.columns[2]]:
        print(f"  {c}: min={out[c].min():.4f} mean={out[c].mean():.4f} max={out[c].max():.4f}")
    if nan_total:
        out = out.interpolate(method="time").ffill().bfill()
        print("NaNs interpolated to reach a gap-free CSV")

    out.round(6).to_csv(OUT)
    print(f"Saved: {OUT} ({OUT.stat().st_size/1e6:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
