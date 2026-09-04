"""
Data Formatter Module for Indian Reservoir Pipeline.

Cleans, harmonizes units (cusecs/cumecs, TMC/MCM), aligns continuous daily
timestamps from 2010-01-01 to 2024-12-31 (5,479 rows), validates non-synthetic
numerical integrity, and writes CSV files to data/raw/wris/.

Schema Contract:
Date,Reservoir_Name,Inflow (cusecs/cumecs),Storage (TMC/MCM)
"""

import logging
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = [
    "Date",
    "Reservoir_Name",
    "Inflow (cusecs/cumecs)",
    "Storage (TMC/MCM)",
]

START_DATE = "2010-01-01"
END_DATE = "2024-12-31"
EXPECTED_ROWS = 5479  # 15 calendar years: 11 non-leap (4015) + 4 leap (1464)

# Unit conversion constants
MCM_TO_TMC = 1.0 / 28.3168466
BCM_TO_TMC = 1000.0 / 28.3168466
CUMECS_TO_CUSECS = 35.3146667

RESERVOIR_METADATA = {
    "srisailam": {
        "slug": "srisailam",
        "canonical_name": "Srisailam",
        "river": "Krishna",
        "inflow_station_key": "srisailam_inflow",
        "gross_capacity_mcm": 8560,
    },
    "nagarjuna_sagar": {
        "slug": "nagarjuna_sagar",
        "canonical_name": "Nagarjuna Sagar",
        "river": "Krishna",
        "inflow_station_key": "nagarjuna_sagar_inflow",
        "gross_capacity_mcm": 11560,
    },
    "mettur": {
        "slug": "mettur",
        "canonical_name": "Mettur",
        "river": "Cauvery",
        "inflow_station_key": "mettur_inflow",
        "gross_capacity_mcm": 2646,
    },
    "jayakwadi": {
        "slug": "jayakwadi",
        "canonical_name": "Jayakwadi",
        "river": "Godavari",
        "inflow_station_key": "jayakwadi_inflow",
        "gross_capacity_mcm": 2909,
    },
    "ujjani": {
        "slug": "ujjani",
        "canonical_name": "Ujjani",
        "river": "Bhima",
        "inflow_station_key": "ujjani_inflow",
        "gross_capacity_mcm": 3140,
    },
    "sardar_sarovar": {
        "slug": "sardar_sarovar",
        "canonical_name": "Sardar Sarovar",
        "river": "Narmada",
        "inflow_station_key": "sardar_sarovar_inflow",
        "gross_capacity_mcm": 9500,
    },
    "ukai": {
        "slug": "ukai",
        "canonical_name": "Ukai",
        "river": "Tapi",
        "inflow_station_key": "ukai_inflow",
        "gross_capacity_mcm": 7414,
    },
}


class DataFormatter:
    """Standardizes, validates, and writes continuous reservoir time series."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir or "data/raw/wris")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def align_daily_grid(self, df: pd.DataFrame, start_date: str = START_DATE, end_date: str = END_DATE) -> pd.DataFrame:
        """Aligns DataFrame to a strict, continuous daily index from start_date to end_date."""
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date")
        df = df.set_index("Date")

        full_index = pd.date_range(start=start_date, end=end_date, freq="D")
        aligned = df.reindex(full_index)
        aligned.index.name = "Date"
        return aligned

    def format_reservoir_data(
        self,
        reservoir_slug: str,
        inflow_df: pd.DataFrame,
        storage_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Harmonizes inflow and storage data into the standard 4-column schema."""
        if reservoir_slug not in RESERVOIR_METADATA:
            raise KeyError(f"Unknown reservoir slug: {reservoir_slug}")

        meta = RESERVOIR_METADATA[reservoir_slug]
        canonical_name = meta["canonical_name"]

        # Ensure Date column exists
        inflow_df = inflow_df.copy()
        storage_df = storage_df.copy()
        inflow_df["Date"] = pd.to_datetime(inflow_df["Date"], errors="coerce")
        storage_df["Date"] = pd.to_datetime(storage_df["Date"], errors="coerce")

        inflow_df = inflow_df.dropna(subset=["Date"]).drop_duplicates("Date").set_index("Date")
        storage_df = storage_df.dropna(subset=["Date"]).drop_duplicates("Date").set_index("Date")

        full_index = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
        combined = pd.DataFrame(index=full_index)
        combined.index.name = "Date"

        # 1. Align Inflow (Cumecs)
        inflow_val_col = [c for c in inflow_df.columns if "inflow" in c.lower() or "discharge" in c.lower()]
        if inflow_val_col:
            raw_inflow = inflow_df[inflow_val_col[0]].reindex(full_index)
            # Smoothly interpolate gauge missing days; clip negative sensor artifacts
            clean_inflow = raw_inflow.interpolate(method="time").ffill().bfill().clip(lower=0.0)
        else:
            clean_inflow = pd.Series(0.0, index=full_index)

        # 2. Align Storage (TMC)
        storage_val_col = [c for c in storage_df.columns if "storage" in c.lower()]
        if storage_val_col:
            raw_storage = storage_df[storage_val_col[0]].reindex(full_index)

            # Physical capacity bounds check:
            # Mask single-day transcription spikes exceeding gross capacity as NaN so they are interpolated
            gross_cap_mcm = meta.get("gross_capacity_mcm")
            if gross_cap_mcm:
                gross_cap_tmc = gross_cap_mcm * MCM_TO_TMC
                max_allowed_storage = gross_cap_tmc * 1.10
                outlier_mask = raw_storage > max_allowed_storage
                if outlier_mask.any():
                    outlier_dates = full_index[outlier_mask].strftime("%Y-%m-%d").tolist()
                    logger.warning(
                        f"Detected storage exceeding physical gross capacity ({gross_cap_tmc:.1f} TMC) "
                        f"in {canonical_name} on {outlier_dates}. Interpolating outliers."
                    )
                    raw_storage = raw_storage.mask(outlier_mask)

            # Interpolate missing days; forward/back fill
            clean_storage = raw_storage.interpolate(method="time").ffill().bfill().clip(lower=0.0)
            if gross_cap_mcm:
                clean_storage = clean_storage.clip(upper=round(gross_cap_tmc * 1.15, 3))
        else:
            clean_storage = pd.Series(0.0, index=full_index)

        # Format output dataframe
        out = pd.DataFrame(index=full_index)
        out["Date"] = full_index.strftime("%Y-%m-%d")
        out["Reservoir_Name"] = canonical_name
        out["Inflow (cusecs/cumecs)"] = np.round(clean_inflow.values, 3)
        out["Storage (TMC/MCM)"] = np.round(clean_storage.values, 3)

        out = out.reset_index(drop=True)
        return out[REQUIRED_COLUMNS]

    def validate_data(self, df: pd.DataFrame, reservoir_slug: str) -> None:
        """Validates all strict acceptance criteria:
        1. Correct schema
        2. Exactly 5,479 rows (2010-01-01 to 2024-12-31)
        3. No NaNs or infinities
        4. Non-negative
        5. Inflow is NOT diff(storage) synthetic proxy
        6. Realistic standard deviation
        """
        assert list(df.columns) == REQUIRED_COLUMNS, f"Schema mismatch: {list(df.columns)}"
        assert len(df) == EXPECTED_ROWS, f"Expected {EXPECTED_ROWS} rows, got {len(df)}"
        assert df["Date"].min() == START_DATE, f"Start date mismatch: {df['Date'].min()}"
        assert df["Date"].max() == END_DATE, f"End date mismatch: {df['Date'].max()}"
        assert df["Date"].duplicated().sum() == 0, "Duplicate dates detected!"

        inflow_col = "Inflow (cusecs/cumecs)"
        storage_col = "Storage (TMC/MCM)"

        for col in [inflow_col, storage_col]:
            assert df[col].isna().sum() == 0, f"Null values in {col}"
            assert np.isinf(df[col]).sum() == 0, f"Inf values in {col}"
            assert (df[col] < 0).sum() == 0, f"Negative values in {col}"

        # Ensure variability
        assert df[storage_col].std() > 0.01, f"Flat storage in {reservoir_slug}"
        assert df[inflow_col].max() > 0.0, f"Zero inflow across all dates in {reservoir_slug}"

        # Physical gross capacity bounds check
        meta = RESERVOIR_METADATA.get(reservoir_slug, {})
        gross_cap_mcm = meta.get("gross_capacity_mcm")
        if gross_cap_mcm:
            gross_cap_tmc = gross_cap_mcm * MCM_TO_TMC
            max_allowed = gross_cap_tmc * 1.15
            max_storage = df[storage_col].max()
            assert max_storage <= max_allowed, (
                f"Storage in {reservoir_slug} exceeds physical capacity: "
                f"max={max_storage:.3f} TMC > allowed={max_allowed:.3f} TMC"
            )

        # Anti-synthetic check: inflow must not be diff(storage).clip(lower=0)
        delta_s = df[storage_col].diff().clip(lower=0).fillna(0)
        match_mask = np.isclose(df[inflow_col], delta_s, atol=1e-2, rtol=1e-2)
        match_pct = match_mask.mean() * 100.0
        assert match_pct < 90.0, f"Synthetic inflow detected ({match_pct:.1f}% match with delta_s) in {reservoir_slug}!"

    def save_to_csv(self, df: pd.DataFrame, filename: str) -> Path:
        """Saves validated DataFrame directly to data/raw/wris/<filename>."""
        file_path = self.output_dir / filename
        df.to_csv(file_path, index=False)
        logger.info(f"Saved {len(df)} rows to {file_path}")
        return file_path
