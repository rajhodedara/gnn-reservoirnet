import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\odeda\Desktop\Projects\PBL")
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.reservoir_pipeline.nwdp_extractor import NWDPExtractor  # noqa: E402

ex = NWDPExtractor(cache_dir=str(PROJECT_ROOT / "data" / "raw" / "nwdp_cache"))

CHECKS = [
    ("1b9088b5-d196-4c5d-8780-a888e7e9e86b", "WADENEPALLY", "nagarjuna_sagar"),
    ("9c659865-ab21-4ffa-a3f9-edbae14f5c86", "KOPERGAON", "jayakwadi"),
    ("fca9df0b-47b1-4f1a-8e59-1b43a8c0ae73", "BILIGUNDULU", "mettur"),
    ("5708264d-5aea-4e39-8e64-e837f55d4c1b", "MANDLESHWAR", "sardar_sarovar"),
]

for rid, station, slug in CHECKS:
    print(f"=== {station} @ {rid[:8]} (target: {slug}) ===", flush=True)
    recs = ex.fetch_station_records(rid, station, max_records=60000)
    print(f"records: {len(recs)}", flush=True)
    if not recs:
        continue
    df = pd.DataFrame(recs)
    vcol = next(c for c in df.columns if "discharge" in c.lower())
    df["date"] = pd.to_datetime(df["Data Acquisition Time"], dayfirst=True, errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    df["_v"] = pd.to_numeric(df[vcol], errors="coerce")
    print(f"range: {df['date'].min().date()} .. {df['date'].max().date()} | non-null: {int(df['_v'].notna().sum())}")
    # sample 3 dates per year against the patched CSV
    patched = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "wris_v2" / f"{slug}.csv")
    patched["Date"] = pd.to_datetime(patched["Date"])
    patched = patched.set_index("Date")["Inflow (cusecs/cumecs)"]
    sample_dates = [f"{y}-08-15" for y in (2015, 2019, 2022, 2024)]
    for d in sample_dates:
        ts = pd.Timestamp(d)
        live = df.loc[df["date"] == ts, "_v"]
        live_v = float(live.iloc[0]) if len(live) else None
        csv_v = float(patched.get(ts, float("nan")))
        match = "" if (live_v is None) else ("MATCH" if abs(live_v - csv_v) < 0.01 else "DIFF")
        print(f"  {d}: live={live_v} csv={csv_v:.3f} {match}")
    print(flush=True)
