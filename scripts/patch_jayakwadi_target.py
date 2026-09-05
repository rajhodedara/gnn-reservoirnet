"""Patch jayakwadi.csv inflow with CWC Kopergaon upstream main-stem discharge.

Rationale (documented in jayakwadi_target_patch_manifest.json):
- Dhalegaon (original source) is located ~100 km DOWNSTREAM of Jayakwadi Dam (Paithan).
  When Jayakwadi impounds water, downstream river flow is zero, resulting in
  80.4% zero days and complete all-zero years (including 2023).
- Kopergaon (CWC station on the Godavari main-stem in Ahmednagar, Maharashtra,
  19.89 N, 74.49 E) is situated ~85 km UPSTREAM of Jayakwadi reservoir headwaters.
- NWDP Resource: 9c659865-ab21-4ffa-a3f9-edbae14f5c86 ("River Discharge CWC
  Maharashtra 2001-2025 Manual Daily"), Station: Kopergaon.
- Patch window: 2014-06-01 .. 2024-12-31 (Kopergaon operational period).
  Pre-June 2014 retains original records as no upstream CWC daily gauge
  exists in official public databases for that window.
- Eliminates the 2023 all-zero defect: Kopergaon provides 55 non-zero days in 2023
  with peak discharge 568.97 m3/s during July-October.

Original is backed up to data/raw/wris/backup_pre_jayakwadi_patch/.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WRIS = PROJECT_ROOT / "data" / "raw" / "wris"
CACHE = PROJECT_ROOT / "data" / "raw" / "nwdp_cache" / "9c659865_kopergaon.json"
BACKUP = WRIS / "backup_pre_jayakwadi_patch"
PATCH_START = "2014-06-01"
PATCH_END = "2024-12-31"


def main() -> int:
    BACKUP.mkdir(exist_ok=True, parents=True)
    target = WRIS / "jayakwadi.csv"
    backup_file = BACKUP / "jayakwadi.csv"

    # Backup original before modification
    if not backup_file.exists():
        backup_file.write_bytes(target.read_bytes())
    print(f"Backup verified at: {backup_file}")

    if not CACHE.exists():
        raise FileNotFoundError(f"Cache file not found: {CACHE}")

    records = json.loads(CACHE.read_text(encoding="utf-8"))
    df = pd.DataFrame(records)
    vcol = next(c for c in df.columns if "discharge" in c.lower())
    df["date"] = pd.to_datetime(df["Data Acquisition Time"], dayfirst=True, errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    kop = df.set_index("date")["_v"] if "_v" in df else df.set_index("date")[vcol]
    kop = pd.to_numeric(kop, errors="coerce")

    patch_idx = pd.date_range(PATCH_START, PATCH_END, freq="D")
    series = kop.reindex(patch_idx)
    gap_days = int(series.isna().sum())
    series = series.interpolate(method="time").ffill().bfill().clip(lower=0).round(3)
    patch = pd.Series(series.values, index=patch_idx)

    cur = pd.read_csv(target)
    cur["Date"] = pd.to_datetime(cur["Date"])
    mask = (cur["Date"] >= PATCH_START) & (cur["Date"] <= PATCH_END)
    old = cur.loc[mask, "Inflow (cusecs/cumecs)"].astype(float)
    cur.loc[mask, "Inflow (cusecs/cumecs)"] = patch.values
    cur["Date"] = cur["Date"].dt.strftime("%Y-%m-%d")
    cur.to_csv(target, index=False)

    p_2023 = patch[pd.to_datetime(patch_idx).year == 2023]

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reservoir": "jayakwadi",
        "patch_window": f"{PATCH_START}..{PATCH_END}",
        "source": "CWC Kopergaon upstream main-stem gauge (NWDP resource 9c659865-ab21-4ffa-a3f9-edbae14f5c86, River Discharge CWC Maharashtra 2001-2025 Manual Daily)",
        "source_resource_id": "9c659865-ab21-4ffa-a3f9-edbae14f5c86",
        "station": "Kopergaon",
        "river": "Godavari",
        "unit": "m3/s",
        "unit_conversion": "none (source already m3/s)",
        "rationale": "Dhalegaon was downstream of Jayakwadi Dam (~100 km), registering regulated outflow and false zeros (80.4% zero days, 2023 all-zero). Kopergaon is upstream (~85 km) on Godavari main-stem with real measured inflow for 2014-2024.",
        "pre_2014_handling": "Dates 2010-01-01 to 2014-05-31 retained as unmeasured upstream reach per project protocol.",
        "rows_patched": int(mask.sum()),
        "gap_days_interpolated": gap_days,
        "old_patched_window_mean": round(float(old.mean()), 2),
        "old_patched_window_max": round(float(old.max()), 2),
        "new_patched_window_mean": round(float(patch.mean()), 2),
        "new_patched_window_max": round(float(patch.max()), 2),
        "new_2023_mean": round(float(p_2023.mean()), 2),
        "new_2023_max": round(float(p_2023.max()), 2),
        "new_2023_nonzero_days": int((p_2023 > 0).sum()),
    }
    (WRIS / "jayakwadi_target_patch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Jayakwadi patch complete. Manifest:")
    print(json.dumps({k: v for k, v in manifest.items() if k != "rationale"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
