"""Patch almatti / krishnaraja_sagara / tungabhadra inflow in data/raw/wris/
with the official Karnataka WRD/KSNDMC daily dam-inflow series
(.cluster/inflow-data/karnataka_man_reservoir_data.csv, units: cusecs).

Originals are backed up to data/raw/wris/backup_pre_ka_patch/ before writing.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WRIS = PROJECT_ROOT / "data" / "raw" / "wris"
KA_FILE = PROJECT_ROOT / ".cluster" / "inflow-data" / "karnataka_man_reservoir_data.csv"
BACKUP = WRIS / "backup_pre_ka_patch"
CUSECS_TO_CUMECS = 1 / 35.3146667

DAM_MAP = {
    "Almatti Dam": "almatti",
    "K.R.Sagara Dam": "krishnaraja_sagara",
    "Tungabhadra Dam": "tungabhadra",
}

START, END = "2010-01-01", "2024-12-31"
FULL_IDX = pd.date_range(START, END, freq="D")


def main() -> int:
    BACKUP.mkdir(exist_ok=True)
    df = pd.read_csv(KA_FILE, usecols=lambda c: c.strip() in ("Reservoir Name", "Monitoring Date", "Inflow (Cusecs)"))
    df.columns = [c.strip() for c in df.columns]
    df["date"] = pd.to_datetime(df["Monitoring Date"], dayfirst=True, errors="coerce").dt.normalize()
    df = df.dropna(subset=["date"])
    df["inflow_cusecs"] = pd.to_numeric(df["Inflow (Cusecs)"], errors="coerce")

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source_file": str(KA_FILE),
        "source_description": "Karnataka WRD/KSNDMC daily manual reservoir monitoring (official at-dam inflow, cusecs)",
        "conversion": "cusecs / 35.3146667 -> m3/s (cumecs)",
        "reservoirs": {},
    }

    for dam, slug in DAM_MAP.items():
        sub = df[df["Reservoir Name"].str.strip() == dam].sort_values("date").drop_duplicates("date", keep="last")
        s = sub.set_index("date")["inflow_cusecs"].clip(lower=0)
        s = s.reindex(FULL_IDX)
        gap_days = int(s.isna().sum())
        s = s.interpolate(method="time").ffill().bfill()
        cumecs = (s * CUSECS_TO_CUMECS).round(3)

        target = WRIS / f"{slug}.csv"
        target_backup = BACKUP / f"{slug}.csv"
        if not target_backup.exists():
            target_backup.write_bytes(target.read_bytes())

        cur = pd.read_csv(target)
        cur["Date"] = pd.to_datetime(cur["Date"])
        old = cur["Inflow (cusecs/cumecs)"].astype(float)
        cur = cur[cur["Date"].dt.year.between(2010, 2024)].reset_index(drop=True)
        old = old.iloc[: len(cur)]
        new = cumecs.values

        cur["Inflow (cusecs/cumecs)"] = new
        cur["Date"] = cur["Date"].dt.strftime("%Y-%m-%d")
        cur.to_csv(target, index=False)

        old_i, new_i = pd.Series(old), pd.Series(new)
        jjas_new = new_i[pd.to_datetime(cur["Date"]).dt.month.isin([6, 7, 8, 9])]
        manifest["reservoirs"][slug] = {
            "dam_name": dam,
            "ksndmc_rows_raw": int(len(sub)),
            "gap_days_interpolated": gap_days,
            "old_mean_cumecs": round(float(old_i.mean()), 2),
            "new_mean_cumecs": round(float(new_i.mean()), 2),
            "new_max_cumecs": round(float(new_i.max()), 2),
            "new_zero_pct": round(float((new_i == 0).mean() * 100), 1),
            "new_jjas_share_pct": round(float(jjas_new.sum() / new_i.sum() * 100), 1),
            "new_2024_mean_cumecs": round(float(new_i[pd.to_datetime(cur['Date']).dt.year == 2024].mean()), 2),
        }
        m = manifest["reservoirs"][slug]
        print(f"{slug:<20} old_mean={m['old_mean_cumecs']:>9.2f}  new_mean={m['new_mean_cumecs']:>9.2f}  "
              f"new_max={m['new_max_cumecs']:>10.2f}  zero%={m['new_zero_pct']:>5.1f}  "
              f"JJAS%={m['new_jjas_share_pct']:>5.1f}  2024_mean={m['new_2024_mean_cumecs']:>9.2f}  gaps={gap_days}")

    (WRIS / "ka_inflow_patch_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\nPatch manifest: {WRIS / 'ka_inflow_patch_manifest.json'}")
    print(f"Backups: {BACKUP}")
    return 0


if __name__ == "__main__":
    main()
