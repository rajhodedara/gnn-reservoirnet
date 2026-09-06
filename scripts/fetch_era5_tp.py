r"""Download ERA5 hourly total precipitation (tp) per year -> merge -> daily sums.

Uses the standard reanalysis-era5-single-levels request (guaranteed valid form),
calendar-correct day lists, all 24 hours. Then computes daily sums locally:

    data/raw/era5/era5_tp_peninsular_2010_2024.nc   (daily sums, mm/day)
Requires C:\Users\odeda\.cdsapirc (CDS personal access token).
"""

import calendar
import sys
from pathlib import Path

import cdsapi
import xarray as xr

PROJECT_ROOT = Path(r"C:\Users\odeda\Desktop\Projects\PBL")
ERA5 = PROJECT_ROOT / "data" / "raw" / "era5"
YEARS = list(range(2010, 2025))


def year_days(year: int) -> list[str]:
    # CDS rejects lists with repeated values; per-month validity is handled server-side
    return [f"{d:02d}" for d in range(1, 32)]


def fetch_year(c: cdsapi.Client, year: int) -> list[Path]:
    out1 = ERA5 / f"era5_tp_hourly_peninsular_{year}_s1.nc"
    out2 = ERA5 / f"era5_tp_hourly_peninsular_{year}_s2.nc"
    
    parts = []
    
    # Semester 1
    if out1.exists() and out1.stat().st_size > 10_000:
        print(f"  {year} S1: already fetched ({out1.stat().st_size/1e6:.1f} MB)", flush=True)
        parts.append(out1)
    else:
        req1 = {
            "product_type": ["reanalysis"],
            "variable": ["total_precipitation"],
            "year": [str(year)],
            "month": [f"{m:02d}" for m in range(1, 7)],
            "day": year_days(year),
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": [23, 73, 8, 85],
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        print(f"  {year} S1: submitting...", flush=True)
        c.retrieve("reanalysis-era5-single-levels", req1).download(str(out1))
        parts.append(out1)
        
    # Semester 2
    if out2.exists() and out2.stat().st_size > 10_000:
        print(f"  {year} S2: already fetched ({out2.stat().st_size/1e6:.1f} MB)", flush=True)
        parts.append(out2)
    else:
        req2 = {
            "product_type": ["reanalysis"],
            "variable": ["total_precipitation"],
            "year": [str(year)],
            "month": [f"{m:02d}" for m in range(7, 13)],
            "day": year_days(year),
            "time": [f"{h:02d}:00" for h in range(24)],
            "area": [23, 73, 8, 85],
            "data_format": "netcdf",
            "download_format": "unarchived",
        }
        print(f"  {year} S2: submitting...", flush=True)
        c.retrieve("reanalysis-era5-single-levels", req2).download(str(out2))
        parts.append(out2)

    return parts


def main() -> int:
    c = cdsapi.Client()
    files = []
    failures = []
    for y in YEARS:
        try:
            files.extend(fetch_year(c, y))
        except Exception as e:
            print(f"  {y}: FAILED {type(e).__name__}: {str(e)[:200]}", flush=True)
            failures.append(y)
    if failures:
        print(f"FAILED years: {failures}", flush=True)
        if len(failures) == len(YEARS):
            return 1
    files = [f for f in files if f.exists()]

    print("Computing daily sums from hourly tp...", flush=True)
    ds = xr.open_mfdataset(files, combine="by_coords")
    tp_daily = ds["tp"].resample(valid_time="D").sum() * 1000.0  # m -> mm/day
    tp_daily = tp_daily.rename("rainfall_mm")
    merged = ERA5 / "era5_tp_peninsular_2010_2024.nc"
    if merged.exists():
        merged.unlink()
    tp_daily.to_netcdf(merged)
    print(f"Saved daily sums: {merged} ({merged.stat().st_size/1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
