"""Patch mettur.csv inflow with the CWC Biligundulu border-station discharge.

Rationale (documented in mettur_target_patch_manifest.json):
- Kodumudi (original source) sits DOWNSTREAM of Mettur dam + Bhavani confluence:
  its series contains dam-release extremes (max 113,410 m3/s in 2024) that are
  dam operations, not reservoir inflow.
- Biligundulu (CWC station on the Karnataka/TN border) measures the Cauvery
  flow arriving toward Mettur: continuous daily record 2001-2025 (NWDP resource
  fca9df0b-47b1-4f1a-8e59-1b43a8c0ae73).
- Patch window: 2010-01-01 .. 2024-12-31 (full 15-year unified Biligundulu series).
- Outlier handling: On 2018-08-19, the raw CWC record is 74,713.0 m3/s (cusecs
  mislabeled as cumecs). This single anomaly is handled by interpolating between
  adjacent days (2018-08-18: 5,250.7 m3/s and 2018-08-20: 4,712.6 m3/s -> 4,981.65 m3/s)
  and clipping at 15,000 m3/s.

Original is backed up to data/raw/wris/backup_pre_mettur_patch/.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WRIS = PROJECT_ROOT / "data" / "raw" / "wris"
CACHE = PROJECT_ROOT / "data" / "raw" / "nwdp_cache" / "fca9df0b_biligundulu.json"
BACKUP = WRIS / "backup_pre_mettur_patch"
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"


def main() -> int:
    BACKUP.mkdir(exist_ok=True, parents=True)
    target = WRIS / "mettur.csv"
    backup_file = BACKUP / "mettur_pre_patch.csv"
    backup_legacy = BACKUP / "mettur.csv"

    # Ensure pre-patch backups exist
    if not backup_file.exists():
        if backup_legacy.exists():
            backup_file.write_bytes(backup_legacy.read_bytes())
        else:
            backup_file.write_bytes(target.read_bytes())
    if not backup_legacy.exists():
        backup_legacy.write_bytes(backup_file.read_bytes())
    print(f"Backup verified at: {backup_file} and {backup_legacy}")

    if not CACHE.exists():
        raise FileNotFoundError(f"Cache file not found: {CACHE}")

    records = json.loads(CACHE.read_text(encoding="utf-8"))
    df = pd.DataFrame(records)
    vcol = next(c for c in df.columns if "discharge" in c.lower())
    df["date"] = pd.to_datetime(df["Data Acquisition Time"], dayfirst=True, errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")
    bil = df.set_index("date")["_v"] if "_v" in df else df.set_index("date")[vcol]
    bil = pd.to_numeric(bil, errors="coerce")

    patch_idx = pd.date_range(START_DATE, END_DATE, freq="D")
    series = bil.reindex(patch_idx)

    # Handle 2018-08-19 single-day outlier (74713.0 m3/s cusecs mislabel)
    if "2018-08-19" in series.index:
        raw_outlier = float(series.loc["2018-08-19"])
        print(f"Handling 2018-08-19 outlier: raw={raw_outlier}")
        series.loc["2018-08-19"] = float("nan")

    gap_days = int(series.isna().sum())
    series = series.interpolate(method="time").ffill().bfill().clip(lower=0, upper=15000).round(3)
    patch = pd.Series(series.values, index=patch_idx)

    cur = pd.read_csv(target)
    cur["Date"] = pd.to_datetime(cur["Date"])
    mask = (cur["Date"] >= START_DATE) & (cur["Date"] <= END_DATE)
    old = cur.loc[mask, "Inflow (cusecs/cumecs)"].astype(float)
    cur.loc[mask, "Inflow (cusecs/cumecs)"] = patch.values
    cur["Date"] = cur["Date"].dt.strftime("%Y-%m-%d")
    cur.to_csv(target, index=False)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reservoir": "mettur",
        "patch_window": f"{START_DATE}..{END_DATE}",
        "source": "CWC Biligundulu border station (NWDP resource fca9df0b-47b1-4f1a-8e59-1b43a8c0ae73, River Discharge CWC Tamil Nadu 2001-2025 Manual Daily)",
        "source_resource_id": "fca9df0b-47b1-4f1a-8e59-1b43a8c0ae73",
        "station": "BILIGUNDULU",
        "river": "Cauvery",
        "unit": "m3/s",
        "unit_conversion": "none (source already m3/s)",
        "rationale": "Kodumudi was downstream of Mettur dam + Bhavani confluence containing dam-release extremes up to 113,410 m3/s; Biligundulu measures Cauvery main-stem inflow arriving at Mettur across the full 2010-2024 window (5,322 valid days, 97.1% coverage).",
        "outlier_handling": "2018-08-19 value 74713.0 m3/s (cusecs mislabeled as cumecs) interpolated between 2018-08-18 (5250.7) and 2018-08-20 (4712.6) -> 4981.65 m3/s; clipped at 15000 m3/s.",
        "rows_patched": int(mask.sum()),
        "gap_days_interpolated": gap_days,
        "old_mean": round(float(old.mean()), 2),
        "old_max": round(float(old.max()), 2),
        "new_mean": round(float(patch.mean()), 2),
        "new_max": round(float(patch.max()), 2),
        "new_2024_mean": round(float(patch[pd.to_datetime(patch_idx).year == 2024].mean()), 2),
        "new_2024_max": round(float(patch[pd.to_datetime(patch_idx).year == 2024].max()), 2),
    }
    (WRIS / "mettur_target_patch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Mettur patch complete. Manifest:")
    print(json.dumps({k: v for k, v in manifest.items() if k != "rationale" and k != "outlier_handling"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
