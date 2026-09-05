"""Patch sardar_sarovar.csv inflow with the CWC Mandleshwar main-stem discharge.

Rationale (documented in sardar_sarovar_target_patch_manifest.json):
- Garudeshwar (current source) sits just DOWNSTREAM of the Sardar Sarovar dam:
  its series measures regulated releases, not reservoir inflow.
- Mandleshwar (CWC station on the Narmada main stem, Madhya Pradesh) measures
  the Narmada flow upstream of the SSP dam chain: 100% complete daily record
  2021-2025 (1,826 rows, max 52,000 m3/s in the 2023 Narmada flood).
- Patch window: 2021-01-01 .. 2024-12-31 (Mandleshwar availability).
  2010-2020 keeps Garudeshwar, flagged as release-contaminated.
- Known caveat: Omkareshwar / Indira Sagar reservoirs lie between Mandleshwar
  and SSP — some regulation between gauge and dam is not captured.

Original is backed up to data/raw/wris/backup_pre_ssp_patch/.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\odeda\Desktop\Projects\PBL")
WRIS = PROJECT_ROOT / "data" / "raw" / "wris"
CACHE = PROJECT_ROOT / "data" / "raw" / "nwdp_cache" / "e62e4559_mandleshwar.json"
BACKUP = WRIS / "backup_pre_ssp_patch"
BREAK_DATE = "2021-01-01"


def main() -> int:
    BACKUP.mkdir(exist_ok=True)
    target = WRIS / "sardar_sarovar.csv"
    backup_file = BACKUP / "sardar_sarovar.csv"
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
        "reservoir": "sardar_sarovar",
        "patch_window": "2021-01-01..2024-12-31",
        "source": "CWC Mandleshwar main-stem gauge (NWDP resource e62e4559-8d96-49d1-91e8-29f0750c0324, velocity+discharge 2021-25)",
        "rationale": "Garudeshwar is just downstream of the SSP dam: measures regulated releases, not inflow. Mandleshwar measures the Narmada upstream of the SSP dam chain (1,826 complete daily rows 2021-2025, max 52,000 m3/s in the 2023 flood).",
        "unit_conversion": "none (source already m3/s)",
        "rows_patched": int(mask.sum()),
        "gap_days_interpolated": gap_days,
        "old_2021_2024_mean": round(float(old.mean()), 2),
        "new_2024_mean": round(float(patch[pd.to_datetime(patch_idx).year == 2024].mean()), 2),
        "new_2024_max": round(float(patch[pd.to_datetime(patch_idx).year == 2024].max()), 2),
        "known_caveat": "Omkareshwar/Indira Sagar reservoirs lie between Mandleshwar and SSP: some inter-dam regulation not captured. 2010-2020 remains Garudeshwar (release-contaminated); 2021 source break documented here.",
    }
    (WRIS / "sardar_sarovar_target_patch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "rationale" and k != "known_caveat"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
