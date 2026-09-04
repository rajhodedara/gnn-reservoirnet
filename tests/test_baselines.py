"""Tests for the baselines (persistence, climatology, metrics)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_baselines as rb  # noqa: E402


def _daily():
    idx = pd.date_range("2020-01-01", "2023-12-31", freq="D")
    rng = np.random.default_rng(1)
    return pd.DataFrame({"A": rng.uniform(10, 100, len(idx))}, index=idx)


class TestPersistence:
    def test_equals_last_week_sum(self):
        daily = _daily()
        persist = rb.persistence_forecast(daily)
        t = pd.Timestamp("2021-06-15")
        expected = daily.loc[t - pd.Timedelta(days=6): t, "A"].sum()
        assert persist.loc[t, "A"] == pytest.approx(expected)

    def test_target_is_next7_sum(self):
        daily = _daily()
        targets = rb.weekly_sum_next7(daily)
        t = pd.Timestamp("2021-06-15")
        expected = daily.loc[t + pd.Timedelta(days=1): t + pd.Timedelta(days=7), "A"].sum()
        assert targets.loc[t, "A"] == pytest.approx(expected)


class TestClimatology:
    def test_ignores_post_train_data(self):
        """Test-year extremes must not change the climatology forecast."""
        daily = _daily()
        targets = rb.weekly_sum_next7(daily)
        train_end_year = 2022
        clim_train = rb.climatology_forecast(daily, targets, train_end_year=train_end_year)
        # Corrupt the test-year inflow with absurd values and recompute
        daily_bad = daily.copy()
        daily_bad.loc["2023-01-01":] = 1e9
        targets_bad = rb.weekly_sum_next7(daily_bad)
        clim_bad = rb.climatology_forecast(daily_bad, targets_bad, train_end_year=train_end_year)
        common = clim_train.loc["2023-01-01":].index
        assert np.allclose(clim_train.loc[common].values, clim_bad.loc[common].values, atol=1e-6)


class TestMetrics:
    def test_nse_perfect_and_known(self):
        obs = pd.DataFrame({"A": [1.0, 2.0, 3.0, 4.0]})
        perfect = obs.copy()
        m = rb.metrics_table(obs, perfect)
        assert m["A"]["NSE"] == pytest.approx(1.0)
        obs_mean = pd.DataFrame({"A": [2.5, 2.5, 2.5, 2.5]})
        m2 = rb.metrics_table(obs, obs_mean)
        assert m2["A"]["NSE"] == pytest.approx(0.0)
