"""Regression tests for the committed ERA5 point-extraction CSV."""

from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV = PROJECT_ROOT / "data" / "raw" / "era5" / "reservoir_era5_daily.csv"


@pytest.mark.skipif(not CSV.exists(), reason="reservoir_era5_daily.csv not generated yet")
class TestEra5PointExtraction:
    def test_file_shape_and_window(self):
        df = pd.read_csv(CSV, parse_dates=["Date"], index_col="Date")
        assert df.shape[0] == 5479  # 2010-01-01..2024-12-31 daily
        assert df.index.min().strftime("%Y-%m-%d") == "2010-01-01"
        assert df.index.max().strftime("%Y-%m-%d") == "2024-12-31"

    def test_all_reservoirs_and_features_present(self):
        df = pd.read_csv(CSV)
        slugs = ["almatti", "tungabhadra", "krishnaraja_sagara", "mettur",
                 "nagarjuna_sagar", "srisailam", "jayakwadi", "ujjani",
                 "sardar_sarovar", "ukai"]
        for s in slugs:
            for feat in ["runoff", "evap", "soil_moisture"]:
                assert f"{s}_{feat}" in df.columns, f"missing {s}_{feat}"

    def test_no_nans_and_physical_ranges(self):
        df = pd.read_csv(CSV)
        assert not df.isna().any().any()
        runoff = df.filter(like="_runoff")
        assert (runoff >= 0).all().all()  # surface runoff cannot be negative
        soil = df.filter(like="_soil_moisture")
        assert (soil > 0).all().all() and (soil < 1).all().all()  # volumetric m3/m3
