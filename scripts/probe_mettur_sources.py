import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\odeda\Desktop\Projects\PBL")
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.reservoir_pipeline.nwdp_extractor import NWDPExtractor  # noqa: E402

ex = NWDPExtractor(cache_dir=str(PROJECT_ROOT / "data" / "raw" / "nwdp_cache"))
TN_CWC = "fca9df0b-47b1-4f1a-8e59-1b43a8c0ae73"   # CWC TN 2001-2025 (Kodumudi source)
KA_CWC = "f95150ea-c8fc-4740-8815-d9c34c9d53a3"   # CWC KA 2001-2025 (Huvinhedigi/Yadgir source)
TN_SWGW = "79c5b4a5-d20c-486e-a249-9d9227648976"  # TN SW&GW dept 2001-2025

VARIANTS = ["THOPPUR", "BILIGUNDULAN", "BILLIGUNDULAN", "BILIGUNDLU", "BILIGUNDL"]

def summarize(records, label):
    df = pd.DataFrame(records)
    vcol = next((c for c in df.columns if "discharge" in c.lower()), None)
    df["date"] = pd.to_datetime(df["Data Acquisition Time"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    df["_v"] = pd.to_numeric(df[vcol], errors="coerce") if vcol else float("nan")
    v = df["_v"]
    print(f"--- {label}: {len(records)} rows, {df['date'].min().date()} .. {df['date'].max().date()}")
    for yr in range(2010, 2025):
        ydf = df[df["date"].dt.year == yr]
        print(f"  {yr}: rows={len(ydf):>4} nn={int(ydf['_v'].notna().sum()):>4} mean={ydf['_v'].mean():>8.1f} max={ydf['_v'].max():>9.1f}")
    return df

# 1. THOPPUR (downstream fallback)
recs = ex.fetch_station_records(TN_CWC, "THOPPUR")
print(f"\n[1] THOPPUR @ TN_CWC: {len(recs)} records")
if recs:
    summarize(recs, "THOPPUR")

# 2. Biligundlu spelling variants on both CWC resources
print()
for res_id, res_label in [(TN_CWC, "TN_CWC"), (KA_CWC, "KA_CWC")]:
    for v in VARIANTS[1:]:
        recs = ex.fetch_station_records(res_id, v)
        if recs:
            print(f"[2] {v} @ {res_label}: {len(recs)} records")
            summarize(recs, f"{v}@{res_label}")
        else:
            print(f"[2] {v} @ {res_label}: 0")

# 3. Scan TN SWGW resource for METTUR stations
print(f"\n[3] Scanning TN_SWGW ({TN_SWGW}) for station names containing METTUR...")
found_records = []
offset = 0
scanned = 0
station_names = set()
while scanned < 120_000:
    batch = ex.query_datastore(resource_id=TN_SWGW, limit=10_000, offset=offset)
    recs = batch.get("result", {}).get("records", [])
    total = batch.get("result", {}).get("total", 0)
    if not recs:
        break
    df_b = pd.DataFrame(recs)
    st_col = next(c for c in df_b.columns if "station" in c.lower())
    station_names.update(df_b[st_col].dropna().astype(str).str.strip().unique())
    m = df_b[st_col].astype(str).str.upper().str.contains("METTUR")
    if m.any():
        found_records.extend(df_b[m].to_dict("records"))
    scanned += len(recs)
    offset += len(recs)
    if offset >= total:
        break

print(f"scanned {scanned} rows; distinct stations: {len(station_names)}")
mettur_like = sorted({n for n in station_names if "METTUR" in n.upper()})
print("METTUR-like stations:", mettur_like if mettur_like else "NONE")
if found_records:
    pd.DataFrame(found_records).to_json(
        str(PROJECT_ROOT / "data" / "raw" / "nwdp_cache" / "79c5b4a5_mettur_rows.json"), indent=1
    )
    print(f"cached {len(found_records)} mettur rows -> 79c5b4a5_mettur_rows.json")
