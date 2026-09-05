import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\odeda\Desktop\Projects\PBL")
sys.path.insert(0, str(PROJECT_ROOT))
from src.data.reservoir_pipeline.nwdp_extractor import NWDPExtractor  # noqa: E402

ex = NWDPExtractor(cache_dir=str(PROJECT_ROOT / "data" / "raw" / "nwdp_cache"))

RESOURCES = [
    ("b8189c98-9066-4f48-a545-168d7c398fc0", "CWC TN velocity+discharge 2021-25"),
    ("d1bf65de-1010-47e5-beb6-23ff7ac82888", "KA WRD discharge gauge 2021-25"),
]

for rid, label in RESOURCES:
    print(f"=== {label} ({rid[:8]}) ===")
    names, dmin, dmax = set(), None, None
    offset, scanned = 0, 0
    total = None
    while scanned < 150_000:
        batch = ex.query_datastore(resource_id=rid, limit=10_000, offset=offset)
        recs = batch.get("result", {}).get("records", [])
        total = batch.get("result", {}).get("total", total)
        if not recs:
            break
        dfb = pd.DataFrame(recs)
        st_col = next(c for c in dfb.columns if "station" in c.lower())
        t_col = next(c for c in dfb.columns if "time" in c.lower() or "date" in c.lower())
        names.update(dfb[st_col].dropna().astype(str).str.strip().unique())
        d = pd.to_datetime(dfb[t_col], dayfirst=True, errors="coerce").dropna()
        if len(d):
            dmin = d.min() if dmin is None else min(dmin, d.min())
            dmax = d.max() if dmax is None else max(dmax, d.max())
        scanned += len(recs)
        offset += len(recs)
        if total and offset >= total:
            break
    print(f"scanned={scanned} range={dmin}..{dmax}")
    interesting = sorted(n for n in names if any(k in n.lower() for k in ("mettur", "cauvery", "kaveri", "biligund", "kodumudi", "bhavani", "musiri", "hogenakkal")))
    print(f"stations ({len(names)}): interesting={interesting}")
    print("sample:", sorted(names)[:15])
    print()
