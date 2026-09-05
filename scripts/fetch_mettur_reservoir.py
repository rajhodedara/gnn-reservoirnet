import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\odeda\Desktop\Projects\PBL")
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.reservoir_pipeline.nwdp_extractor import NWDPExtractor

ex = NWDPExtractor(cache_dir=str(PROJECT_ROOT / "data" / "raw" / "nwdp_cache"))
records = ex.fetch_station_records(
    "fca9df0b-47b1-4f1a-8e59-1b43a8c0ae73",  # River Discharge CWC TN (2001-2025) Manual Daily
    "Mettur Reservoir",
)
print(f"fetched/cached records: {len(records)}")
if not records:
    sys.exit(1)

df = pd.DataFrame(records)
vcol = "Manual Daily River Water Discharge (m3/sec)"
df["date"] = pd.to_datetime(df["Data Acquisition Time"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
df["_v"] = pd.to_numeric(df[vcol], errors="coerce")
print(f"date range: {df['date'].min().date()} .. {df['date'].max().date()} | rows: {len(df)}")
print("value stats:", df["_v"].describe().round(2).to_dict())
print("negative values:", int((df["_v"] < 0).sum()))
print("rows/year (2010-2024):")
for yr in range(2010, 2025):
    ydf = df[df["date"].dt.year == yr]
    print(f"  {yr}: rows={len(ydf):>4} non-null={int(ydf['_v'].notna().sum()):>4} mean={ydf['_v'].mean():.1f} max={ydf['_v'].max():.1f}")
