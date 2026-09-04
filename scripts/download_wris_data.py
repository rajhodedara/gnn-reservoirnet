"""
download_wris_data.py — Multi-source reservoir data acquisition.

Tries multiple sources in priority order:
  1. Local files (data/raw/wris/ already populated)
  2. Kaggle dataset (when running on Kaggle)
  3. NWIC National Water Data Portal (CSV/API)
  4. Dataful.in (manual download helper)
  5. Synthetic fallback (generate_synthetic_wris.py)

Usage:
  python scripts/download_wris_data.py                # auto-detect best source
  python scripts/download_wris_data.py --source nwic   # force NWIC download
  python scripts/download_wris_data.py --source synthetic  # generate synthetic
  python scripts/download_wris_data.py --status        # check what data exists
  python scripts/download_wris_data.py --push-kaggle YOUR_USERNAME  # upload to Kaggle
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("wris")

# ---------------------------------------------------------------------------
# Paths & Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOCAL_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "wris"
KAGGLE_INPUT_DIRS = [
    Path("/kaggle/input/wris-reservoir-data"),
    Path("/kaggle/input/india-reservoir-data"),
    Path("/kaggle/input/cwc-reservoir-data"),
]

KEEP_COLS = ["Date", "inflow", "outflow", "storage"]

# Reservoir config: id → display name
RESERVOIRS = {
    "nagarjuna_sagar": "Nagarjuna Sagar",
    "srisailam": "Srisailam",
    "almatti": "Almatti",
    "tungabhadra": "Tungabhadra",
    "ujjani": "Ujjani",
    "mettur": "Mettur",
    "krishnaraja_sagara": "Krishnaraja Sagara",
    "jayakwadi": "Jayakwadi",
    "sardar_sarovar": "Sardar Sarovar",
    "ukai": "Ukai",
}


# ---------------------------------------------------------------------------
# Source 1: Local files
# ---------------------------------------------------------------------------
def load_local(data_dir: Path | None = None) -> dict[str, pd.DataFrame]:
    """Load existing CSVs from data/raw/wris/."""
    data_dir = data_dir or LOCAL_DATA_DIR
    result = {}
    for rid in RESERVOIRS:
        csv_path = data_dir / f"{rid}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if "Date" in df.columns and len(df) > 0:
                result[rid] = df
                log.info("  [local] %-25s %6d rows", rid, len(df))
    return result


# ---------------------------------------------------------------------------
# Source 2: Kaggle dataset input
# ---------------------------------------------------------------------------
def load_kaggle_input() -> dict[str, pd.DataFrame]:
    """Load from /kaggle/input/ directories (when running on Kaggle)."""
    for kaggle_dir in KAGGLE_INPUT_DIRS:
        if kaggle_dir.exists():
            csvs = list(kaggle_dir.glob("*.csv"))
            if csvs:
                log.info("  [kaggle] Found %d CSVs in %s", len(csvs), kaggle_dir)
                result = {}
                for csv_path in csvs:
                    rid = csv_path.stem
                    if rid in RESERVOIRS:
                        df = pd.read_csv(csv_path)
                        if "Date" in df.columns:
                            result[rid] = df
                            log.info("  [kaggle] %-25s %6d rows", rid, len(df))
                return result
    return {}


# ---------------------------------------------------------------------------
# Source 3: NWIC National Water Data Portal
# ---------------------------------------------------------------------------
def download_nwic(
    output_dir: Path,
    start_date: str = "2005-01-01",
    end_date: str = "2026-12-31",
    timeout: int = 45,
    target_reservoirs: set[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Download from NWIC API (nwdp.nwic.gov.in).

    NWIC provides 500+ APIs. We try the reservoir storage level endpoint.
    Falls back to the India-WRIS API if NWIC fails.
    """
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    output_dir.mkdir(parents=True, exist_ok=True)

    # Build session with retry
    session = requests.Session()
    adapter = HTTPAdapter(
        max_retries=Retry(total=3, backoff_factor=2.0,
                          status_forcelist=[429, 500, 502, 503, 504],
                          allowed_methods=["POST", "GET"])
    )
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"User-Agent": "Mozilla/5.0", "Accept": "application/json"})

    # Column aliases from WRIS/NWIC API responses
    col_aliases = {
        "date": "Date", "readingdate": "Date", "leveldate": "Date", "reading_date": "Date",
        "inflow": "inflow", "dailyinflow": "inflow",
        "outflow": "outflow", "dailyoutflow": "outflow",
        "storage": "storage", "livestorage": "storage", "livecapacity": "storage",
        "currentlivestorage": "storage", "currentlivestorage_bmc": "storage",
    }

    # The API includes suffixes like " Reservoir" and alternate names
    api_name_mapping = {
        "nagarjuna_sagar": "Nagarjuna Sagar",
        "srisailam": "Srisailam Reservoir",
        "almatti": "Almatti Reservoir",
        "tungabhadra": "Tungabhadra Reservoir",
        "ujjani": "Bhima\\Ujjani Reservoir",
        "mettur": "Mettur Reservoir",
        "krishnaraja_sagara": "Krishnaraja Sagar",
        "jayakwadi": "Jayakwadi\\Nath Sagar",
        "sardar_sarovar": "Sardar Sarovar",
        "ukai": "Ukai Reservoir",
    }
    target_reservoirs = target_reservoirs or set(RESERVOIRS.keys())
    res_lookup = {name.lower(): rid for rid, name in api_name_mapping.items() if rid in target_reservoirs}
    buckets: dict[str, list[dict]] = {rid: [] for rid in target_reservoirs}

    # Use India-WRIS API (most tested endpoint)
    api_url = "https://indiawris.gov.in/Dataset/Reservoir"
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    window_days = 365
    window_start = start
    total_windows = ((end - start).days // window_days) + 1
    window_num = 0

    while window_start < end:
        window_end = min(window_start + timedelta(days=window_days - 1), end)
        window_num += 1
        s = window_start.strftime("%Y-%m-%d")
        e = window_end.strftime("%Y-%m-%d")
        log.info("  [nwic] [%d/%d] %s → %s", window_num, total_windows, s, e)

        params = {
            "stateName": "0", "districtName": "0", "agencyName": "CWC",
            "startdate": s, "enddate": e, "download": "false",
            "page": 0, "size": 100000,
        }
        try:
            resp = session.post(api_url, params=params, timeout=timeout)
            resp.raise_for_status()
            records = resp.json().get("data", [])
        except Exception as exc:
            log.warning("  [nwic] Request failed: %s", exc)
            window_start = window_end + timedelta(days=1)
            continue

        if not records:
            window_start = window_end + timedelta(days=1)
            continue

        for rec in records:
            name_lower = str(rec.get("reservoirName", "")).lower()
            if name_lower in res_lookup:
                buckets[res_lookup[name_lower]].append(rec)

        window_start = window_end + timedelta(days=1)

    # Convert to DataFrames
    result = {}
    for rid, raw in buckets.items():
        if not raw:
            continue
        df = pd.DataFrame(raw)
        rename = {col: col_aliases[col.lower()] for col in df.columns if col.lower() in col_aliases}
        df = df.rename(columns=rename)
        cols = [c for c in KEEP_COLS if c in df.columns]
        if "Date" not in cols:
            continue
        df = df[cols]
        df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="last")
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
        for c in ["inflow", "outflow", "storage"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.reset_index(drop=True)
        out = output_dir / f"{rid}.csv"
        try:
            df.to_csv(out, index=False)
            log.info("  [nwic] ✓ %-25s %6d rows", rid, len(df))
        except PermissionError:
            fallback = out.with_name(f"{rid}_backup.csv")
            df.to_csv(fallback, index=False)
            log.warning("  [nwic] ⚠ %-25s %6d rows (Saved to %s due to PermissionError)", rid, len(df), fallback.name)
        result[rid] = df

    return result


# ---------------------------------------------------------------------------
# Source 4: Manual download helper (Dataful.in / NWIC portal)
# ---------------------------------------------------------------------------
def print_manual_download_guide() -> None:
    """Print URLs and instructions for manual CSV download."""
    guide = """
================================================================
       MANUAL DOWNLOAD GUIDE -- Indian Reservoir Data
================================================================

Since automated downloads are slow/blocked, download CSVs manually:

--- Option A: NWIC National Water Data Portal (BEST) -----------
  URL: https://nwdp.nwic.gov.in/
  1. Go to Datasets -> search "Reservoir Water Storage Level (Manual - Daily)"
  2. Filter by state: Telangana, AP, Karnataka, Tamil Nadu, Gujarat, Maharashtra
  3. Click "Download" -> CSV format
  4. Save each CSV to: data/raw/wris/

--- Option B: Dataful.in (PRE-CLEANED) -------------------------
  URL: https://dataful.in/
  1. Search "Water Storage" or "Reservoir"
  2. Download CSV/Parquet for reservoir-wise weekly storage
  3. Note: This is WEEKLY data -- may need interpolation for daily

--- Option C: APWRIMS (AP/Telangana reservoirs) -----------------
  URL: https://apwrims.ap.gov.in/
  -> For: Nagarjuna Sagar, Srisailam
  -> Has daily inflow + outflow + storage

--- Option D: India-WRIS Portal (SLOW but complete) -------------
  URL: https://indiawris.gov.in/
  1. Go to "Reservoir" module
  2. Select reservoir -> date range -> Export CSV
  3. Warning: Very slow, may timeout on large ranges

----------------------------------------------------------------

After downloading, ensure each CSV has columns:
  Date, storage  (minimum)
  Date, inflow, outflow, storage  (ideal)

Save as: data/raw/wris/<reservoir_id>.csv
"""
    for rid, name in RESERVOIRS.items():
        guide += f"  {rid}.csv  ← {name}\n"

    print(guide)


# ---------------------------------------------------------------------------
# Source 5: Synthetic generation
# ---------------------------------------------------------------------------
def generate_synthetic(output_dir: Path, start_year: int = 2005, end_year: int = 2026) -> dict[str, pd.DataFrame]:
    """Generate realistic synthetic data as a last-resort fallback."""
    output_dir.mkdir(parents=True, exist_ok=True)

    date_range = pd.date_range(start=f"{start_year}-01-01", end=f"{end_year}-12-31", freq="D")
    t = np.arange(len(date_range))
    annual_cycle = np.sin(2 * np.pi * t / 365.25 - np.pi / 2)  # Peak in monsoon

    result = {}
    for rid, name in RESERVOIRS.items():
        np.random.seed(hash(rid) % 2**31)  # Deterministic per reservoir

        base_storage = np.random.uniform(500, 2000)
        base_flow = np.random.uniform(50, 200)

        inflow = np.maximum(0, base_flow + base_flow * 2 * annual_cycle + np.random.normal(0, base_flow / 2, len(t)))
        outflow = np.maximum(0, base_flow + base_flow * 1.5 * annual_cycle + np.random.normal(0, base_flow / 3, len(t)))

        storage = np.zeros(len(t))
        storage[0] = base_storage
        for i in range(1, len(t)):
            storage[i] = np.clip(
                storage[i - 1] + (inflow[i] - outflow[i]),
                base_storage * 0.1, base_storage * 2.0,
            )

        df = pd.DataFrame({
            "Date": date_range.strftime("%Y-%m-%d"),
            "inflow": np.round(inflow, 2),
            "outflow": np.round(outflow, 2),
            "storage": np.round(storage, 2),
        })

        out = output_dir / f"{rid}.csv"
        df.to_csv(out, index=False)
        result[rid] = df
        log.info("  [synthetic] %-25s %6d rows", rid, len(df))

    return result


# ---------------------------------------------------------------------------
# Status check
# ---------------------------------------------------------------------------
def check_status(data_dir: Path | None = None) -> None:
    """Print what data exists and what's missing."""
    data_dir = data_dir or LOCAL_DATA_DIR

    print(f"\n{'Reservoir':<28} {'File':<12} {'Rows':>8} {'Cols':>25} {'Date Range':>25}")
    print("-" * 100)

    found = 0
    for rid, name in RESERVOIRS.items():
        csv_path = data_dir / f"{rid}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            cols = ", ".join(df.columns)
            date_range = ""
            if "Date" in df.columns and len(df) > 0:
                date_range = f"{df['Date'].iloc[0]} -> {df['Date'].iloc[-1]}"
            print(f"  + {name:<25} {csv_path.name:<12} {len(df):>8} {cols:>25} {date_range:>25}")
            found += 1
        else:
            print(f"  - {name:<25} {'MISSING':<12}")

    print("-" * 100)
    print(f"  {found}/{len(RESERVOIRS)} reservoirs have data in {data_dir}\n")


# ---------------------------------------------------------------------------
# Kaggle push
# ---------------------------------------------------------------------------
def push_to_kaggle(username: str, data_dir: Path | None = None) -> None:
    """Upload data/raw/wris/ as a Kaggle dataset."""
    data_dir = data_dir or LOCAL_DATA_DIR

    csvs = list(data_dir.glob("*.csv"))
    if not csvs:
        log.error("No CSVs found in %s — download data first.", data_dir)
        sys.exit(1)

    slug = f"{username}/wris-reservoir-data"
    meta = {
        "title": "India WRIS Daily Reservoir Data",
        "id": slug,
        "licenses": [{"name": "CC0-1.0"}],
    }
    meta_path = data_dir / "dataset-metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    log.info("Pushing %d files to kaggle: %s", len(csvs), slug)
    try:
        # Try create first
        r = subprocess.run(
            ["kaggle", "datasets", "create", "-p", str(data_dir), "--dir-mode", "zip"],
            capture_output=True, text=True, check=False,
        )
        if r.returncode != 0 and "already exists" in (r.stdout + r.stderr).lower():
            subprocess.run(
                ["kaggle", "datasets", "version", "-p", str(data_dir),
                 "-m", f"Update {datetime.now():%Y-%m-%d}", "--dir-mode", "zip"],
                check=True,
            )
        log.info("✓ https://www.kaggle.com/datasets/%s", slug)
    except FileNotFoundError:
        log.error("'kaggle' CLI not found. Install: pip install kaggle")
    finally:
        meta_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
def acquire_data(
    source: str | None = None,
    output_dir: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Try sources in priority order until we have data for all reservoirs."""
    output_dir = output_dir or LOCAL_DATA_DIR
    on_kaggle = Path("/kaggle/working").exists()

    # Priority chain
    if source:
        sources = ["local", source] if source != "local" else ["local"]
    else:
        sources = ["local", "kaggle", "nwic", "guide"] if on_kaggle else ["local", "nwic", "guide"]

    result: dict[str, pd.DataFrame] = {}
    missing = set(RESERVOIRS.keys())

    for src in sources:
        if not missing:
            break

        log.info("Trying source: %s (need %d more reservoirs)", src, len(missing))

        if src == "local":
            loaded = load_local(output_dir)

        elif src == "kaggle":
            loaded = load_kaggle_input()
            # Copy Kaggle input to working dir
            if loaded:
                output_dir.mkdir(parents=True, exist_ok=True)
                for rid, df in loaded.items():
                    df.to_csv(output_dir / f"{rid}.csv", index=False)

        elif src == "nwic":
            loaded = download_nwic(output_dir, target_reservoirs=missing)

        elif src == "guide":
            if missing:
                log.info("Automated sources couldn't get all data. Showing manual guide...")
                print_manual_download_guide()
            loaded = {}

        elif src == "synthetic":
            log.warning("Generating SYNTHETIC data for %d missing reservoirs", len(missing))
            loaded = generate_synthetic(output_dir)

        else:
            log.error("Unknown source: %s", src)
            loaded = {}

        # Merge new data
        for rid, df in loaded.items():
            if rid in missing:
                result[rid] = df
                missing.discard(rid)

    # Write manifest
    manifest = {
        "created_at": datetime.now().isoformat(),
        "reservoirs": {
            rid: {
                "rows": len(df),
                "columns": list(df.columns),
                "date_min": str(df["Date"].min()) if "Date" in df.columns else None,
                "date_max": str(df["Date"].max()) if "Date" in df.columns else None,
            }
            for rid, df in result.items()
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Summary
    log.info("=" * 55)
    total = sum(len(df) for df in result.values())
    log.info("RESULT: %d/%d reservoirs, %d total rows", len(result), len(RESERVOIRS), total)
    for rid, df in sorted(result.items()):
        rows = len(df)
        date_range = ""
        if "Date" in df.columns and rows > 0:
            date_range = f"{df['Date'].iloc[0]} → {df['Date'].iloc[-1]}"
        log.info("  %-25s %6d rows  %s", rid, rows, date_range)
    if missing:
        log.warning("MISSING: %s", ", ".join(sorted(missing)))
    log.info("=" * 55)

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-source reservoir data acquisition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Sources (in priority order):
  local      Load existing CSVs from data/raw/wris/
  kaggle     Load from Kaggle dataset input (on Kaggle only)
  nwic       Download from NWIC/WRIS API (run locally — blocked on cloud)
  guide      Print manual download URLs (Dataful.in, NWIC, APWRIMS)
  synthetic  Generate realistic synthetic data (last resort)

Examples:
  python scripts/download_wris_data.py                  # auto: try all sources
  python scripts/download_wris_data.py --source nwic    # force NWIC API
  python scripts/download_wris_data.py --source synthetic
  python scripts/download_wris_data.py --status         # check existing data
  python scripts/download_wris_data.py --guide          # print download URLs
  python scripts/download_wris_data.py --push-kaggle YOUR_USERNAME
        """,
    )
    parser.add_argument("action", nargs="?", default=None, help="Legacy action (e.g. 'scrape')")
    parser.add_argument("--source", choices=["local", "kaggle", "nwic", "guide", "synthetic"],
                        help="Force a specific data source")
    parser.add_argument("--status", action="store_true", help="Check existing data status")
    parser.add_argument("--guide", action="store_true", help="Print manual download guide")
    parser.add_argument("--push-kaggle", metavar="USERNAME", help="Push data to Kaggle dataset")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")

    args = parser.parse_args()

    # Legacy alias support
    if args.action == "scrape" and args.source is None:
        args.source = "nwic"

    if args.status:
        check_status(args.output)
        return

    if args.guide:
        print_manual_download_guide()
        return

    if args.push_kaggle:
        push_to_kaggle(args.push_kaggle, args.output)
        return

    result = acquire_data(source=args.source, output_dir=args.output)
    if not result:
        log.error("No data acquired from any source!")
        sys.exit(1)


if __name__ == "__main__":
    main()
