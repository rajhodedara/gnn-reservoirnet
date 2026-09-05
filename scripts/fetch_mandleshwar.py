import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\odeda\Desktop\Projects\PBL")
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.reservoir_pipeline.nwdp_extractor import NWDPExtractor  # noqa: E402

ex = NWDPExtractor(cache_dir=str(PROJECT_ROOT / "data" / "raw" / "nwdp_cache"))
records = ex.fetch_station_records("e62e4559-8d96-49d1-91e8-29f0750c0324", "Mandleshwar", max_records=60000)
print(f"Mandleshwar records: {len(records)}")
if not records:
    sys.exit(1)

df = pd.DataFrame(records)
vcol = next(c for c in df.columns if "discharge" in c.lower())
df["date"] = pd.to_datetime(df["Data Acquisition Time"], dayfirst=True, errors="coerce").dt.normalize()
df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
df["_v"] = pd.to_numeric(df[vcol], errors="coerce")
print(f"value column: {vcol!r}")
print(f"date range: {df['date'].min().date()} .. {df['date'].max().date()}")
print("value stats:", df["_v"].describe().round(2).to_dict())
print("negatives:", int((df["_v"] < 0).sum()))
for yr in range(2021, 2025):
    ydf = df[df["date"].dt.year == yr]
    print(f"  {yr}: rows={len(ydf):>4} non-null={int(ydf['_v'].notna().sum()):>4} mean={ydf['_v'].mean():>8.1f} max={ydf['_v'].max():>9.1f}")
