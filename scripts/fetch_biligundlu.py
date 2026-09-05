import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\odeda\Desktop\Projects\PBL")
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.reservoir_pipeline.nwdp_extractor import NWDPExtractor  # noqa: E402

pk = json.loads((PROJECT_ROOT / ".agents" / "teamwork_preview_explorer_survey_3" / "nwdp_state_packages.json").read_text(encoding="utf-8"))

out = []
def find(o):
    if isinstance(o, dict):
        name = str(o.get("name", ""))
        if "discharge" in name.lower() and "karnataka" in name.lower():
            out.append({"name": name, "id": o.get("id"), "title": o.get("title")})
        for v in o.values():
            find(v)
    elif isinstance(o, list):
        for v in o:
            find(v)
find(pk)
seen = set()
ka_discharge = []
for p in out:
    if p["name"] not in seen:
        seen.add(p["name"])
        ka_discharge.append(p)
print("Karnataka discharge packages:")
for p in ka_discharge:
    print(" ", p)

# Pick the 2001-2025 manual daily one (covers our window)
target = next((p for p in ka_discharge if "2001" in p["name"] and "2025" in p["name"]), None)
if not target or not target["id"]:
    print("No usable 2001-2025 Karnataka discharge resource id found.")
    sys.exit(1)

print(f"\nFetching stations from: {target['name']} ({target['id']})")
ex = NWDPExtractor(cache_dir=str(PROJECT_ROOT / "data" / "raw" / "nwdp_cache"))
records = ex.fetch_station_records(target["id"], "BILIGUNDLU", max_records=60000)
print(f"BILIGUNDLU records: {len(records)}")
if records:
    df = pd.DataFrame(records)
    vcol = next((c for c in df.columns if "discharge" in c.lower()), None)
    df["date"] = pd.to_datetime(df["Data Acquisition Time"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    df["_v"] = pd.to_numeric(df[vcol], errors="coerce") if vcol else float("nan")
    print(f"date range: {df['date'].min().date()} .. {df['date'].max().date()}")
    print("value stats:", df["_v"].describe().round(2).to_dict())
    for yr in range(2010, 2025):
        ydf = df[df["date"].dt.year == yr]
        print(f"  {yr}: rows={len(ydf):>4} non-null={int(ydf['_v'].notna().sum()):>4} mean={ydf['_v'].mean():.1f} max={ydf['_v'].max():.1f}")
