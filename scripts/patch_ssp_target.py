"""Patch sardar_sarovar.csv inflow with the CWC Mandleshwar main-stem discharge.

Rationale (documented in sardar_sarovar_target_patch_manifest.json):
- Garudeshwar (original source) sits just DOWNSTREAM of the Sardar Sarovar dam:
  its series measures regulated releases, not reservoir inflow.
- Mandleshwar (CWC station on the Narmada main stem, Madhya Pradesh) measures
  the Narmada flow upstream of the SSP dam chain: continuous daily record
  2001-2025 (NWDP resource 5708264d-5aea-4e39-8e64-e837f55d4c1b).
- Patch window: 2010-01-01 .. 2024-12-31 (full 15-year unified Mandleshwar series).
- Known caveat: Omkareshwar / Indira Sagar reservoirs lie upstream of Mandleshwar
  on the Narmada cascade.

Original is backed up to data/raw/wris/backup_pre_ssp_patch/.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WRIS = PROJECT_ROOT / "data" / "raw" / "wris"
CACHE = PROJECT_ROOT / "data" / "raw" / "nwdp_cache" / "5708264d_mandleshwar.json"
BACKUP = WRIS / "backup_pre_ssp_patch"
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"


def main() -> int:
    BACKUP.mkdir(exist_ok=True, parents=True)
    target = WRIS / "sardar_sarovar.csv"
    backup_file = BACKUP / "sardar_sarovar_pre_patch.csv"
    backup_legacy = BACKUP / "sardar_sarovar.csv"

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
    gap_days = int(series.isna().sum())
    series = series.interpolate(method="time").ffill().bfill().clip(lower=0).round(3)
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
        "reservoir": "sardar_sarovar",
        "patch_window": f"{START_DATE}..{END_DATE}",
        "source": "CWC Mandleshwar main-stem gauge (NWDP resource 5708264d-5aea-4e39-8e64-e837f55d4c1b, River Discharge CWC Madhya Pradesh 2001-2025 Manual Daily)",
        "source_resource_id": "5708264d-5aea-4e39-8e64-e837f55d4c1b",
        "station": "Mandleshwar",
        "river": "Narmada",
        "unit": "m3/s",
        "unit_conversion": "none (source already m3/s)",
        "rationale": "Garudeshwar was downstream of the SSP dam and measured regulated releases; Mandleshwar measures Narmada main-stem inflow upstream of the SSP reservoir foreshore across the entire 2010-2024 window (5,356 valid days, 97.8% coverage, zero zeros).",
        "rows_patched": int(mask.sum()),
        "gap_days_interpolated": gap_days,
        "old_mean": round(float(old.mean()), 2),
        "old_max": round(float(old.max()), 2),
        "new_mean": round(float(patch.mean()), 2),
        "new_max": round(float(patch.max()), 2),
        "new_2024_mean": round(float(patch[pd.to_datetime(patch_idx).year == 2024].mean()), 2),
        "new_2024_max": round(float(patch[pd.to_datetime(patch_idx).year == 2024].max()), 2),
        "known_caveat": "Omkareshwar/Indira Sagar reservoirs lie upstream of Mandleshwar; some inter-dam regulation is present, but Mandleshwar provides genuine upstream measured flow into the SSP impoundment reach.",
    }
    (WRIS / "sardar_sarovar_target_patch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Sardar Sarovar patch complete. Manifest:")
    print(json.dumps({k: v for k, v in manifest.items() if k != "rationale" and k != "known_caveat"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
