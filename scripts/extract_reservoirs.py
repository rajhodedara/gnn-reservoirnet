#!/usr/bin/env python
"""
Main Execution Script: Indian Reservoir Historical Data Extraction Pipeline.

Navigates the 4 candidate sources (UW-SASWE/RAT, reservoirs.earth, NWDP, data.gov.in),
extracts genuine historical daily storage and inflow records from NWDP CKAN API / Datastore,
harmonizes units, enforces continuous daily coverage (2010-01-01 to 2024-12-31), validates
non-synthetic numerical integrity, and saves the 7 CSV files to data/raw/wris/.

Usage:
    python scripts/extract_reservoirs.py [options]

Options:
    --output-dir DIR   Directory to write output CSVs (default: data/raw/wris)
    --cache-dir DIR    Directory for NWDP raw cache (default: data/raw/nwdp_cache)
    --skip-probe       Skip live web probing of the 4 candidate sources
    --verify-only      Run validation checks on existing CSVs without re-extracting
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.reservoir_pipeline.source_navigator import SourceNavigator
from src.data.reservoir_pipeline.nwdp_extractor import NWDPExtractor, STATION_RESOURCES
from src.data.reservoir_pipeline.data_formatter import (
    DataFormatter,
    RESERVOIR_METADATA,
    REQUIRED_COLUMNS,
    START_DATE,
    END_DATE,
    EXPECTED_ROWS,
    BCM_TO_TMC,
    MCM_TO_TMC,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("extract_reservoirs")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract continuous daily historical inflow and storage data (2010-2024) for 7 Indian reservoirs."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/wris",
        help="Destination directory for reservoir CSV files (default: data/raw/wris)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="data/raw/nwdp_cache",
        help="Directory to cache raw NWDP API JSON payloads (default: data/raw/nwdp_cache)",
    )
    parser.add_argument(
        "--skip-probe",
        action="store_true",
        help="Skip live network probing of the 4 candidate sources",
    )
    parser.add_argument(
        "--legacy-cache-dir",
        type=str,
        default="data/raw/legacy_cwc_cache",
        help="Directory containing immutable raw CWC storage reference CSVs (default: data/raw/legacy_cwc_cache)",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Run validation checks on existing CSVs without fetching new data",
    )
    return parser.parse_args()


def load_legacy_storage(
    raw_wris_dir: Optional[Path],
    slug: str,
    legacy_cache_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Loads existing raw historical CWC storage series for a reservoir, if available.
    Decoupled from output dir: falls back to immutable legacy_cwc_cache if file does not exist in raw_wris_dir.
    """
    if legacy_cache_dir is None:
        legacy_cache_dir = PROJECT_ROOT / "data" / "raw" / "legacy_cwc_cache"

    legacy_file = None
    if raw_wris_dir and (raw_wris_dir / f"{slug}.csv").exists():
        legacy_file = raw_wris_dir / f"{slug}.csv"
    elif legacy_cache_dir and (legacy_cache_dir / f"{slug}.csv").exists():
        legacy_file = legacy_cache_dir / f"{slug}.csv"

    if legacy_file is None or not legacy_file.exists():
        return pd.DataFrame(columns=["Date", "storage"])

    try:
        df = pd.read_csv(legacy_file)
        if "Date" not in df.columns:
            return pd.DataFrame(columns=["Date", "storage"])

        storage_col = next((c for c in df.columns if "storage" in c.lower()), None)
        if not storage_col:
            return pd.DataFrame(columns=["Date", "storage"])

        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date")
        df["storage_raw"] = pd.to_numeric(df[storage_col], errors="coerce")

        # Check if already in TMC:
        # e.g. "Storage (TMC/MCM)" or "tmc" in storage_col.lower() and "mcm" in storage_col.lower()
        col_lower = storage_col.lower()
        if (
            storage_col == "Storage (TMC/MCM)"
            or ("tmc" in col_lower and "mcm" in col_lower)
            or "tmc" in col_lower
        ):
            df["storage_tmc"] = df["storage_raw"]
        else:
            # In legacy WRIS files, values were stored in BCM (e.g. 1.0 - 6.0 BCM)
            # Convert BCM to TMC: 1 BCM = 1000 MCM / 28.3168466 ≈ 35.31467 TMC
            # If already in MCM (> 50), convert to TMC using MCM_TO_TMC
            mean_val = df["storage_raw"].dropna().mean()
            if mean_val < 20.0:
                df["storage_tmc"] = df["storage_raw"] * BCM_TO_TMC
            else:
                df["storage_tmc"] = df["storage_raw"] * MCM_TO_TMC

        return df[["Date", "storage_tmc"]].rename(columns={"storage_tmc": "storage"})
    except Exception as e:
        logger.warning(f"Failed loading legacy storage for {slug}: {e}")
        return pd.DataFrame(columns=["Date", "storage"])


def run_pipeline(
    output_dir: str,
    cache_dir: str,
    skip_probe: bool = False,
    legacy_cache_dir: Optional[str] = "data/raw/legacy_cwc_cache",
):
    """Executes the end-to-end extraction and harmonization pipeline."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    legacy_path = Path(legacy_cache_dir) if legacy_cache_dir else None

    # -----------------------------------------------------------------
    # 1. Multi-Source Navigation & Verification
    # -----------------------------------------------------------------
    navigator = SourceNavigator()
    if not skip_probe:
        logger.info("Probing the 4 candidate sources...")
        probe_results = navigator.probe_all_sources(timeout=10)
        report = navigator.generate_probe_report(probe_results)
        print("\n" + report + "\n")
    else:
        logger.info("Skipping source probe as requested.")

    # -----------------------------------------------------------------
    # 2. Ingestion from NWDP CKAN API
    # -----------------------------------------------------------------
    logger.info("Initializing NWDP Extractor and querying genuine observed time series...")
    extractor = NWDPExtractor(cache_dir=cache_dir)
    formatter = DataFormatter(output_dir=output_dir)

    # 2.1 Fetch Srisailam Storage from NWDP AP SW Department (Resource be847b75-154e-4cc8-b4ff-f56ad8735644)
    logger.info("Fetching Srisailam daily storage from NWDP AP SW datastore...")
    srisailam_storage_df = extractor.fetch_srisailam_storage_df()
    if not srisailam_storage_df.empty:
        # Convert MCM to TMC: 1 TMC = 28.3168466 MCM
        srisailam_storage_df["storage"] = srisailam_storage_df["storage_mcm"] * MCM_TO_TMC
        srisailam_storage_df = srisailam_storage_df[["Date", "storage"]]
        logger.info(f"Retrieved {len(srisailam_storage_df)} daily storage records for Srisailam.")

    results_summary = []

    for slug, meta in RESERVOIR_METADATA.items():
        logger.info(f"Processing reservoir: {meta['canonical_name']} ({slug})...")

        # 1. Inflow from CWC River Discharge Station on NWDP
        inflow_key = meta["inflow_station_key"]
        inflow_df = extractor.fetch_inflow_df(inflow_key)
        logger.info(f"  Inflow records retrieved: {len(inflow_df)} rows from {inflow_key}")

        # 2. Storage
        if slug == "srisailam" and not srisailam_storage_df.empty:
            # Use NWDP AP SW Storage
            storage_df = srisailam_storage_df.copy()
            # If 2024 is missing from AP SW 1970-2023 set, blend with CWC bulletin / legacy CWC series
            legacy_storage = load_legacy_storage(legacy_path, slug)
            if legacy_storage.empty:
                legacy_storage = load_legacy_storage(out_path, slug)
            if not legacy_storage.empty:
                storage_df["Date"] = pd.to_datetime(storage_df["Date"])
                legacy_storage["Date"] = pd.to_datetime(legacy_storage["Date"])
                merged_storage = pd.merge(
                    pd.DataFrame(index=pd.date_range(START_DATE, END_DATE, freq="D")),
                    storage_df,
                    left_index=True,
                    right_on="Date",
                    how="left",
                )
                # Fill trailing gaps from legacy CWC series
                merged_storage = pd.merge(merged_storage, legacy_storage, on="Date", how="left", suffixes=("", "_leg"))
                merged_storage["storage"] = merged_storage["storage"].combine_first(merged_storage["storage_leg"])
                storage_df = merged_storage[["Date", "storage"]].dropna(subset=["Date"])
        else:
            # Load genuine historical CWC storage series from immutable legacy cache or existing outputs
            storage_df = load_legacy_storage(legacy_path, slug)
            if storage_df.empty:
                storage_df = load_legacy_storage(out_path, slug)

        # 3. Format, Align Continuous Daily Grid (2010-01-01 to 2024-12-31), Harmonize Units
        final_df = formatter.format_reservoir_data(slug, inflow_df, storage_df)

        # 4. Strict Validation
        formatter.validate_data(final_df, slug)

        # 5. Save final CSV to data/raw/wris/<slug>.csv
        out_file = formatter.save_to_csv(final_df, f"{slug}.csv")

        results_summary.append({
            "Reservoir": meta["canonical_name"],
            "File": out_file.name,
            "Rows": len(final_df),
            "Start": final_df["Date"].min(),
            "End": final_df["Date"].max(),
            "Mean_Inflow_cumecs": round(final_df["Inflow (cusecs/cumecs)"].mean(), 2),
            "Max_Inflow_cumecs": round(final_df["Inflow (cusecs/cumecs)"].max(), 2),
            "Mean_Storage_TMC": round(final_df["Storage (TMC/MCM)"].mean(), 2),
        })

    # -----------------------------------------------------------------
    # Print Final Extraction Table
    # -----------------------------------------------------------------
    summary_df = pd.DataFrame(results_summary)
    print("\n" + "=" * 80)
    print(" EXTRACTION COMPLETED SUCCESSFULLY: 7 RESERVOIRS (2010-2024)")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80 + "\n")


def verify_existing_files(output_dir: str):
    """Verifies existing CSV files in data/raw/wris/."""
    formatter = DataFormatter(output_dir=output_dir)
    for slug, meta in RESERVOIR_METADATA.items():
        csv_file = Path(output_dir) / f"{slug}.csv"
        if not csv_file.exists():
            raise FileNotFoundError(f"Missing expected CSV: {csv_file}")
        df = pd.read_csv(csv_file)
        formatter.validate_data(df, slug)
        print(f"[OK] {meta['canonical_name']} ({csv_file.name}): {len(df)} rows, valid and non-synthetic.")
    print("\nAll 7 reservoir CSV files strictly verified!")


def main():
    args = parse_args()
    try:
        if args.verify_only:
            verify_existing_files(args.output_dir)
        else:
            run_pipeline(
                output_dir=args.output_dir,
                cache_dir=args.cache_dir,
                skip_probe=args.skip_probe,
                legacy_cache_dir=args.legacy_cache_dir,
            )
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
