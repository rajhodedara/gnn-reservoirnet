"""Tests for scripts/build_wris_v2.py — storage cleanup logic."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import build_wris_v2 as bwv  # noqa: E402


def make_series(values, start="2010-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="D")
    return pd.Series(np.asarray(values, dtype=float), index=idx)


class TestScaleCorrection:
    def test_bcm_source_rescaled_to_tmc(self):
        """A legacy BCM-scale series (max ~3.1 'units', cap 121.5 TMC) must be
        rescaled x35.31 under the bcm prior, reaching ~90% of capacity."""
        raw = make_series(np.linspace(0.3, 3.1, 500))  # BCM-scale, gentle
        cleaned, stats = bwv.clean_storage(raw, cap_tmc=121.5, prior="bcm_to_tmc")
        assert stats["scale_counts"]["bcm_to_tmc"] == 1
        assert cleaned.max() == pytest.approx(3.1 * bwv.TMC_PER_BCM, rel=0.05)

    def test_tmc_drought_window_not_inflated(self):
        """A TMC series whose storage legitimately stays below 25% of capacity
        (multi-year drought) must NOT be rescaled by the bcm candidate."""
        raw = make_series(np.full(500, 20.0))  # cap 110.9 TMC, 18% — drought
        cleaned, stats = bwv.clean_storage(raw, cap_tmc=110.9, prior="identity")
        assert stats["scale_counts"]["bcm_to_tmc"] == 0
        assert cleaned.max() == pytest.approx(20.0, abs=0.5)

    def test_mcm_mislabeled_segment_rescued(self):
        """Values 28x too large (MCM rows inside a TMC file) violate bounds and
        must be rescued by the mcm_to_tmc scale."""
        vals = np.concatenate([np.full(60, 90.0), np.full(60, 2500.0), np.full(60, 85.0)])
        raw = make_series(vals)
        cleaned, stats = bwv.clean_storage(raw, cap_tmc=100.0, prior="identity")
        mid = cleaned.iloc[60:120]
        assert mid.max() == pytest.approx(2500.0 / bwv.MCM_PER_TMC, rel=0.05)
        assert cleaned.max() <= 100.0


class TestArtifactMasking:
    def test_impossible_jump_masked_and_interpolated(self):
        """A 0 -> 60 TMC one-day jump on a 100 TMC reservoir (>10%/day) must be
        masked and interpolated, leaving no violating pairs."""
        vals = np.concatenate([np.full(100, 40.0), [100.0], np.full(100, 42.0)])
        raw = make_series(vals)
        cleaned, stats = bwv.clean_storage(raw, cap_tmc=100.0, prior="identity")
        d = cleaned.diff().abs().dropna()
        assert (d <= 10.0 + 1e-6).all()
        assert stats["artifact_days_masked"] >= 1

    def test_exact_zero_treated_as_missing(self):
        """Exact-0.0 storage is missing data, never hydrology; the cleaned
        series must not contain 0 between positive anchors."""
        vals = [45.0] * 50 + [0.0] * 20 + [47.0] * 50
        raw = make_series(vals)
        cleaned, stats = bwv.clean_storage(raw, cap_tmc=100.0, prior="identity")
        assert stats["zero_days_converted"] == 20
        assert (cleaned > 0).all()


class TestOutputContract:
    def test_cleaned_series_bounded_and_finite(self):
        rng = np.random.default_rng(42)
        base = 50 + 20 * np.sin(np.linspace(0, 12 * np.pi, 800))
        vals = base + rng.normal(0, 1, 800)
        vals[300] = 500.0  # impossible spike
        vals[500:520] = 0.0  # zero floor
        raw = make_series(vals)
        cleaned, stats = bwv.clean_storage(raw, cap_tmc=100.0, prior="identity")
        assert cleaned.notna().all()
        assert (cleaned >= 0).all() and (cleaned <= 100.0).all()
        assert np.isfinite(cleaned.values).all()

    def test_schema_and_row_count(self):
        """The 4-column schema contract and 5,479-row window are enforced."""
        assert bwv.REQUIRED_COLUMNS == [
            "Date",
            "Reservoir_Name",
            "Inflow (cusecs/cumecs)",
            "Storage (TMC/MCM)",
        ]
        assert bwv.EXPECTED_ROWS == 5479
