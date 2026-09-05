"""Patch mettur.csv inflow with the CWC Biligundulu border-station discharge.

Rationale (documented in mettur_target_patch_manifest.json):
- Kodumudi (current source) sits DOWNSTREAM of Mettur dam + Bhavani confluence:
  its series contains dam-release extremes (max 113,410 m3/s in 2024) that are
  dam operations, not reservoir inflow — and the GNN collapses trying to
  predict them (run #2/#3 Mettur NSE -0.5 .. -1.9).
- Biligundulu (CWC station on the Karnataka/TN border) measures the Cauvery
  flow arriving toward Mettur: 100% complete daily record 2021-2025,
  physically coherent magnitudes (max 4,947 m3/s).
- Patch window: 2021-01-01 .. 2024-12-31 (Biligundulu availability).
  2010-2020 keeps Kodumudi (best available for that era), flagged as
  release-contaminated in the manifest.

Original is backed up to data/raw/wris/backup_pre_mettur_patch/.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WRIS = PROJECT_ROOT / "data" / "raw" / "wris"
CACHE = PROJECT_ROOT / "data" / "raw" / "nwdp_cache" / "b8189c98_biligundulu.json"
BACKUP = WRIS / "backup_pre_mettur_patch"
BREAK_DATE = "2021-01-01"


def main() -> int:
    BACKUP.mkdir(exist_ok=True)
    target = WRIS / "mettur.csv"
    backup_file = BACKUP / "mettur.csv"
    if not backup_file.exists():
        backup_file.write_bytes(target.read_bytes())
    print(f"Backup: {backup_file}")

    records = json.loads(CACHE.read_text(encoding="utf-8"))
    df = pd.DataFrame(records)
    vcol = next(c for c in df.columns if "discharge" in c.lower())
    df["date"] = pd.to_datetime(df["Data Acquisition Time"], dayfirst=True, errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    bil = df.set_index("date")["_v"] if "_v" in df else df.set_index("date")[vcol]
    bil = pd.to_numeric(bil, errors="coerce")

    patch_idx = pd.date_range("2021-01-01", "2024-12-31", freq="D")
    series = bil.reindex(patch_idx)
    gap_days = int(series.isna().sum())
    series = series.interpolate(method="time").ffill().bfill().clip(lower=0).round(3)
    patch = pd.Series(series.values, index=patch_idx)

    cur = pd.read_csv(target)
    cur["Date"] = pd.to_datetime(cur["Date"])
    mask = (cur["Date"] >= BREAK_DATE) & (cur["Date"] <= "2024-12-31")
    old = cur.loc[mask, "Inflow (cusecs/cumecs)"].astype(float)
    cur.loc[mask, "Inflow (cusecs/cumecs)"] = patch.values
    cur["Date"] = cur["Date"].dt.strftime("%Y-%m-%d")
    cur.to_csv(target, index=False)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reservoir": "mettur",
        "patch_window": "2021-01-01..2024-12-31",
        "source": "CWC Biligundulu border station (NWDP resource b8189c98-9066-4f48-a545-168d7c398fc0, velocity+discharge 2021-25)",
        "rationale": "Kodumudi is downstream of Mettur dam + Bhavani confluence: contains dam-release extremes (max 113,410 m3/s in 2024) that are operations, not inflow. Biligundulu measures the Cauvery arriving at Mettur (max 4,947 m3/s).",
        "unit_conversion": "none (source already m3/s)",
        "rows_patched": int(mask.sum()),
        "gap_days_interpolated": gap_days,
        "old_2021_2024_mean": round(float(old.mean()), 2),
        "old_2021_2024_max": round(float(old.max()), 2),
        "new_2024_mean": round(float(patch[pd.to_datetime(patch_idx).year == 2024].mean()), 2),
        "new_2024_max": round(float(patch[pd.to_datetime(patch_idx).year == 2024].max()), 2),
        "known_limitation": "2010-2020 remains Kodumudi (release-contaminated); 2021 source break documented here.",
    }
    (WRIS / "mettur_target_patch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "rationale"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
