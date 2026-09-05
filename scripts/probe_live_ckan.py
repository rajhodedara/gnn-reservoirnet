import json
import sys
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(r"C:\Users\odeda\Desktop\Projects\PBL")
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.reservoir_pipeline.nwdp_extractor import NWDPExtractor  # noqa: E402

import urllib3
urllib3.disable_warnings()

BASE = "https://nwdp.nwic.gov.in/api/3/action/"

print("=== 1. LIVE CKAN package_search for our dams ===")
for q in ["jayakwadi", "paithan", "mettur reservoir", "nagarjuna sagar inflow", "sardar sarovar inflow", "biligundulu"]:
    try:
        r = requests.get(BASE + "package_search", params={"q": q, "rows": 5}, verify=False, timeout=30)
        data = r.json()
        results = data.get("result", {}).get("results", [])
        count = data.get("result", {}).get("count", 0)
        print(f"\nq='{q}': {count} datasets")
        for res in results[:5]:
            print(f"  - {res.get('name')} | {res.get('title')}")

            for rr in res.get("resources", [])[:4]:
                print(f"      resource: {rr.get('id')} | {str(rr.get('name', ''))[:60]} | format={rr.get('format')}")
    except Exception as e:
        print(f"  FAILED: {e}")

print()
print("=== 2. KA 1972-2020 (a21e8e48): station -> River mapping ===")
ex = NWDPExtractor(cache_dir=str(PROJECT_ROOT / "data" / "raw" / "nwdp_cache"))
rid = "a21e8e48-d5e3-4a35-8274-007e52f92daf"
names_rivers, dmin, dmax = {}, None, None
offset, scanned, total = 0, 0, None
while scanned < 400_000:
    batch = ex.query_datastore(resource_id=rid, limit=10_000, offset=offset)
    recs = batch.get("result", {}).get("records", [])
    total = batch.get("result", {}).get("total", total)
    if not recs:
        break
    dfb = pd.DataFrame(recs)
    st_col = next(c for c in dfb.columns if "station" in c.lower())
    t_col = next(c for c in dfb.columns if "time" in c.lower() or "date" in c.lower())
    r_col = next((c for c in dfb.columns if "river" in c.lower() and "local" not in c.lower() and "sub" not in c.lower()), None)
    d = pd.to_datetime(dfb[t_col], dayfirst=True, errors="coerce").dropna()
    if len(d):
        dmin = d.min() if dmin is None else min(dmin, d.min())
        dmax = d.max() if dmax is None else max(dmax, d.max())
    for _, row in dfb.iterrows():
        st = str(row[st_col]).strip()
        rv = str(row[r_col]).strip() if r_col else ""
        if st not in names_rivers:
            names_rivers[st] = {"river": rv, "first": d.min() if len(d) else None, "last": d.max() if len(d) else None, "rows": 0}
        names_rivers[st]["rows"] += 1
        dd = pd.to_datetime([row[t_col]], dayfirst=True, errors="coerce")
        if len(dd) and dd[0] is not pd.NaT:
            if names_rivers[st]["first"] is None or dd[0] < names_rivers[st]["first"]:
                names_rivers[st]["first"] = dd[0]
            if names_rivers[st]["last"] is None or dd[0] > names_rivers[st]["last"]:
                names_rivers[st]["last"] = dd[0]
    scanned += len(recs)
    offset += len(recs)
    if total and offset >= total:
        break

print(f"scanned={scanned} range={dmin}..{dmax} stations={len(names_rivers)}")
cauvery_sites = {k: v for k, v in names_rivers.items() if "cauvery" in v["river"].lower() or "kaveri" in v["river"].lower()}
print(f"Cauvery-basin stations: {len(cauvery_sites)}")
for k, v in cauvery_sites.items():
    print(f"  {k} | river={v['river']} | rows={v['rows']} | {v['first']} .. {v['last']}")
if not cauvery_sites:
    rivers_seen = sorted({v['river'] for v in names_rivers.values()})
    print("  rivers present:", rivers_seen[:30])
