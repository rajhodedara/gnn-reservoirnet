"""Tests for the rule-curve level model and ML-baseline helpers (torch-free parts)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from eval_levels_rulecurve import implied_releases, pooled_nse  # noqa: E402


class TestImpliedReleases:
    def test_recovers_known_release(self):
        # storage rises 10 TMC/day, inflow 0 -> release 0 (filling)
        idx = pd.date_range("2020-01-01", periods=10)
        st = pd.Series(np.arange(10) * 10.0, index=idx)
        inf = pd.Series(0.0, index=idx)
        r = implied_releases(st, inf)
        assert (r.dropna() == 0).all()

    def test_release_positive_when_draining(self):
        idx = pd.date_range("2020-01-01", periods=10)
        st = pd.Series(np.linspace(100, 50, 10), index=idx)  # falling
        inf = pd.Series(0.0, index=idx)
        r = implied_releases(st, inf)
        assert (r.dropna() > 0).all()

    def test_never_negative(self):
        idx = pd.date_range("2020-01-01", periods=20)
        st = pd.Series(np.random.default_rng(0).uniform(0, 100, 20), index=idx)
        inf = pd.Series(np.random.default_rng(1).uniform(0, 50, 20), index=idx)
        assert (implied_releases(st, inf).dropna() >= 0).all()


class TestPooledNse:
    def test_perfect(self):
        o = np.arange(10.0)
        assert np.isclose(pooled_nse(o, o), 1.0)

    def test_ignores_nan(self):
        o = np.array([1.0, 2.0, np.nan, 4.0])
        p = np.array([1.0, 2.0, 99.0, 4.0])
        assert np.isclose(pooled_nse(o, p), 1.0)


def test_ml_baselines_importable_without_running():
    """The script must import (helpers reusable) without heavy side effects."""
    import importlib
    m = importlib.import_module("run_ml_baselines")
    assert hasattr(m, "sarima_forecast")
    assert hasattr(m, "rf_forecast")
    assert hasattr(m, "lstm_forecast")
    assert hasattr(m, "pooled_nse")


def test_weekly_series_resamples_to_weekly():
    import run_ml_baselines as rb
    idx = pd.date_range("2020-01-01", periods=400)
    t = pd.DataFrame({"a": np.arange(400.0)}, index=idx)
    w = rb._weekly_series(t, "a")
    # weekly spacing
    assert (w.index.to_series().diff().dropna() == pd.Timedelta(days=7)).all()
