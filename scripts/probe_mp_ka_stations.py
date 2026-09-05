import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\odeda\Desktop\Projects\PBL")
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.reservoir_pipeline.nwdp_extractor import NWDPExtractor  # noqa: E402

ex = NWDPExtractor(cache_dir=str(PROJECT_ROOT / "data" / "raw" / "nwdp_cache"))

RESOURCES = [
    ("e62e4559-8d96-49d1-91e8-29f0750c0324", "CWC MP velocity+discharge 2021-25"),
    ("a21e8e48-d5e3-4a35-8274-007e52f92daf", "KA WRD discharge gauge 1972-2020"),
]

for rid, label in RESOURCES:
    print(f"=== {label} ({rid[:8]}) ===", flush=True)
    names, rivers, dmin, dmax = set(), {}, None, None
    offset, scanned, total = 0, 0, None
    while scanned < 200_000:
        batch = ex.query_datastore(resource_id=rid, limit=10_000, offset=offset)
        recs = batch.get("result", {}).get("records", [])
        total = batch.get("result", {}).get("total", total)
        if not recs:
            break
        dfb = pd.DataFrame(recs)
        st_col = next(c for c in dfb.columns if "station" in c.lower())
        t_col = next(c for c in dfb.columns if "time" in c.lower() or "date" in c.lower())
        r_col = next((c for c in dfb.columns if "river" in c.lower() and "local" not in c.lower() and "sub" not in c.lower()), None)
        station_names = dfb[st_col].dropna().astype(str).str.strip()
        names.update(station_names.unique())
        if r_col:
            for st, rv in zip(station_names, dfb[r_col].astype(str).str.strip()):
                rivers.setdefault(st, rv)
        d = pd.to_datetime(dfb[t_col], dayfirst=True, errors="coerce").dropna()
        if len(d):
            dmin = d.min() if dmin is None else min(dmin, d.min())
            dmax = d.max() if dmax is None else max(dmax, d.max())
        scanned += len(recs)
        offset += len(recs)
        if total and offset >= total:
            break
    print(f"scanned={scanned} range={dmin}..{dmax} stations={len(names)}", flush=True)
    if rivers:
        for st_lo, rv in rivers.items():
            if any(k in st_lo for k in ["hoshangabad", "mandleshwar", "omkareshwar", "punasa", "sandiya", "narmada"]):
                print(f"  NARMADA SITE: {st_lo} (river={rv})", flush=True)
            if any(k in st_lo for k in ["cauvery", "kaveri", "biligund", "mettur", "k.r.s", "krishnaraja"]):
                print(f"  CAUVERY SITE: {st_lo} (river={rv})", flush=True)
    print("sample stations:", sorted(names)[:20], flush=True)
    print(flush=True)
