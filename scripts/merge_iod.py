"""Merge the IOD (DMI) index into data/raw/enso/combined_climate_indices.csv.

The ReservoirGNN slices climate input as [:, :3] = ENSO (ONI/SOI/NINO34) and
[:, -1:] = IOD. Until now the combined file had only 3 columns, so the model's
"IOD" input was duplicated Nino3.4. This script appends the real monthly DMI
from data/raw/climate_indices/iod.csv, backing up the original first.

Idempotent: re-running refreshes the iod column in place.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COMBINED = PROJECT_ROOT / "data" / "raw" / "enso" / "combined_climate_indices.csv"
IOD = PROJECT_ROOT / "data" / "raw" / "climate_indices" / "iod.csv"
WINDOW_START, WINDOW_END = "2010-01-01", "2024-12-31"


def main() -> int:
    combined = pd.read_csv(COMBINED)
    iod = pd.read_csv(IOD)

    combined["Date"] = pd.to_datetime(combined["Date"])
    iod["Date"] = pd.to_datetime(iod["Date"]).dt.normalize()

    # Monthly alignment: normalize both to month start
    combined["Date"] = combined["Date"].dt.to_period("M").dt.to_timestamp()
    iod["Date"] = iod["Date"].dt.to_period("M").dt.to_timestamp()
    iod = iod.drop_duplicates("Date").set_index("Date")["iod"]

    backup = COMBINED.with_suffix(".backup_noiod.csv")
    if not backup.exists():
        backup.write_bytes(COMBINED.read_bytes())
        print(f"Backup written: {backup}")

    had_iod = "iod" in combined.columns
    merged = combined.merge(iod.rename("iod"), left_on="Date", right_index=True, how="left")

    win = merged[(merged["Date"] >= WINDOW_START) & (merged["Date"] <= WINDOW_END)]
    stats = {
        "rows_total": int(len(merged)),
        "iod_non_null_total": int(merged["iod"].notna().sum()),
        "iod_non_null_in_window": int(win["iod"].notna().sum()),
        "window_rows": int(len(win)),
        "iod_mean": round(float(win["iod"].mean()), 4),
        "iod_std": round(float(win["iod"].std()), 4),
        "iod_min": round(float(win["iod"].min()), 4),
        "iod_max": round(float(win["iod"].max()), 4),
        "refreshed_existing_column": had_iod,
    }

    if stats["iod_non_null_in_window"] != stats["window_rows"]:
        missing = win[win["iod"].isna()]["Date"].dt.strftime("%Y-%m").tolist()
        print(f"ABORT: iod missing for {len(missing)} months in the analysis window: {missing[:12]}...")
        return 1

    merged["Date"] = merged["Date"].dt.strftime("%Y-%m-%d")
    merged.to_csv(COMBINED, index=False)

    print(f"Merged iod into {COMBINED}")
    print(json.dumps(stats, indent=2))
    y2023 = win[win["Date"].dt.year == 2023][["Date", "oni", "iod"]]
    print("2023 monthly ONI vs IOD (El Nino x positive-IOD sanity check):")
    print(y2023.assign(Date=y2023["Date"].dt.strftime("%Y-%m")).to_string(index=False))

    stats["generated_at"] = datetime.now(timezone.utc).isoformat()
    (PROJECT_ROOT / "data" / "raw" / "enso" / "iod_merge_report.json").write_text(
        json.dumps(stats, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
