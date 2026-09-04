"""Regression tests for the IOD merge (data/raw/enso/combined_climate_indices.csv)."""

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.wris_loader import load_climate_indices  # noqa: E402

COMBINED = PROJECT_ROOT / "data" / "raw" / "enso" / "combined_climate_indices.csv"


class TestIodMerge:
    def test_combined_has_iod_column(self):
        df = pd.read_csv(COMBINED)
        assert "iod" in df.columns
        assert len(df) == 948  # no rows lost in the merge

    def test_loader_returns_four_daily_columns(self):
        daily = load_climate_indices(str(COMBINED))
        assert list(daily.columns) == ["ONI", "SOI", "NINO34", "IOD"]
        # Daily frequency, continuous
        assert (daily.index.to_series().diff().dropna() == pd.Timedelta(days=1)).all()

    def test_no_iod_nans_in_analysis_window(self):
        daily = load_climate_indices(str(COMBINED))
        win = daily.loc["2010-01-01":"2024-12-31", "IOD"]
        assert win.notna().all(), f"IOD NaNs in window: {win[win.isna()].index[:5]}"

    def test_2023_el_nino_with_positive_iod(self):
        """2023 super El Nino (ONI > 0.5 Jun-Dec) co-occurs with positive IOD —
        the interaction the design hypothesis needs must be representable."""
        daily = load_climate_indices(str(COMBINED))
        jja_2023 = daily.loc["2023-06-01":"2023-12-31"]
        assert (jja_2023["ONI"] > 0.5).all()
        assert (jja_2023["IOD"] > 0.4).all()
