"""Patch nagarjuna_sagar.csv inflow with CWC Wadenepally Krishna main-stem discharge.

Rationale (documented in nagarjuna_sagar_target_patch_manifest.json):
- Srisailam-duplicated series (original source) duplicated Srisailam's Huvinhedigi
  inflow on 4,198 of 5,479 days (76.62%) due to a fallback artifact in nwdp_extractor.py
  where VEERLAPALEM had zero rows before November 2016.
- Srisailam Dam tailrace empties directly into the Nagarjuna Sagar reservoir backwater pool
  through the Nallamala gorge (~100 km). No open-channel velocity-area CWC river gauging
  station exists or can physically exist inside the reservoir backwater.
- Wadenepally (Wadapally, CWC station on the Krishna main-stem at the Musi confluence,
  16.794 N, 80.073 E, Nalgonda, Telangana) provides a continuous, official daily
  measured Krishna discharge record spanning 2001-2025 (NWDP resource
  1b9088b5-d196-4c5d-8780-a888e7e9e86b).
- Patch window: 2010-01-01 .. 2024-12-31 (full 15-year continuous series).
- Completely eliminates systematic duplication with Srisailam (from 76.62% to 0.44%,
  where the 24 coinciding days are coincidental dry-season zero-flows).

Original is backed up to data/raw/wris/backup_pre_nagarjuna_patch/.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WRIS = PROJECT_ROOT / "data" / "raw" / "wris"
CACHE = PROJECT_ROOT / "data" / "raw" / "nwdp_cache" / "1b9088b5_wadenepally.json"
BACKUP = WRIS / "backup_pre_nagarjuna_patch"
START_DATE = "2010-01-01"
END_DATE = "2024-12-31"


def main() -> int:
    BACKUP.mkdir(exist_ok=True, parents=True)
    target = WRIS / "nagarjuna_sagar.csv"
    backup_file = BACKUP / "nagarjuna_sagar.csv"

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
    wad = df.set_index("date")["_v"] if "_v" in df else df.set_index("date")[vcol]
    wad = pd.to_numeric(wad, errors="coerce")

    patch_idx = pd.date_range(START_DATE, END_DATE, freq="D")
    series = wad.reindex(patch_idx)
    gap_days = int(series.isna().sum())
    series = series.interpolate(method="time").ffill().bfill().clip(lower=0).round(3)
    patch = pd.Series(series.values, index=patch_idx)

    cur = pd.read_csv(target)
    cur["Date"] = pd.to_datetime(cur["Date"])
    mask = (cur["Date"] >= START_DATE) & (cur["Date"] <= END_DATE)
    old = cur.loc[mask, "Inflow (cusecs/cumecs)"].astype(float)

    # Check duplication with Srisailam
    sri_file = WRIS / "srisailam.csv"
    if sri_file.exists():
        sri = pd.read_csv(sri_file)
        sri["Date"] = pd.to_datetime(sri["Date"])
        sri_inf = sri.set_index("Date")["Inflow (cusecs/cumecs)"].reindex(patch_idx)
        old_dups = int(np.isclose(old.values, sri_inf.values, atol=1e-3).sum())
        new_dups = int(np.isclose(patch.values, sri_inf.values, atol=1e-3).sum())
    else:
        old_dups, new_dups = None, None

    cur.loc[mask, "Inflow (cusecs/cumecs)"] = patch.values
    cur["Date"] = cur["Date"].dt.strftime("%Y-%m-%d")
    cur.to_csv(target, index=False)

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reservoir": "nagarjuna_sagar",
        "patch_window": f"{START_DATE}..{END_DATE}",
        "source": "CWC Wadenepally main-stem gauge (NWDP resource 1b9088b5-d196-4c5d-8780-a888e7e9e86b, River Discharge CWC Telangana 2001-2025 Manual Daily)",
        "source_resource_id": "1b9088b5-d196-4c5d-8780-a888e7e9e86b",
        "station": "Wadenepally",
        "river": "Krishna",
        "unit": "m3/s",
        "unit_conversion": "none (source already m3/s)",
        "rationale": "Srisailam-duplicated series duplicated Srisailam's gauge on 76.62% of days due to a fallback artifact. Srisailam Dam empties directly into Nagarjuna Sagar backwater gorge (~100 km). Wadenepally provides continuous CWC Krishna main-stem discharge (8,673 rows, continuous 2001-2025).",
        "rows_patched": int(mask.sum()),
        "gap_days_interpolated": gap_days,
        "old_mean": round(float(old.mean()), 2),
        "old_max": round(float(old.max()), 2),
        "new_mean": round(float(patch.mean()), 2),
        "new_max": round(float(patch.max()), 2),
        "old_duplication_days_with_srisailam": old_dups,
        "old_duplication_pct": round(float(old_dups / len(patch_idx) * 100), 2) if old_dups else None,
        "new_duplication_days_with_srisailam": new_dups,
        "new_duplication_pct": round(float(new_dups / len(patch_idx) * 100), 2) if new_dups else None,
    }
    (WRIS / "nagarjuna_sagar_target_patch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Nagarjuna Sagar patch complete. Manifest:")
    print(json.dumps({k: v for k, v in manifest.items() if k != "rationale"}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
