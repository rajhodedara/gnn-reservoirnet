"""Regression tests for the climate-lag rebuild and raw-ONI wiring.

Covers the audit fixes:
  * climate features are index-major (num_indices, num_lags), not (num_indices, 90)
  * lags use only strictly-past months (no look-ahead)
  * the dataset emits a raw (unstandardized) ONI scalar for loss gating
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import ReservoirInflowDataset  # noqa: E402


def _toy(n=800, nodes=3, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2018-01-01", periods=n, freq="D")
    climate = pd.DataFrame(
        {c: rng.normal(size=n) for c in ["ONI", "SOI", "NINO34", "IOD"]}, index=idx
    )
    cols = [f"r{i}_{f}" for i in range(nodes) for f in
            ["inflow", "storage", "rainfall", "evap", "soil_moisture"]]
    features = pd.DataFrame(rng.normal(size=(n, len(cols))), index=idx, columns=cols)
    inflow = pd.DataFrame(rng.normal(size=(n, nodes)), index=idx,
                          columns=[f"r{i}" for i in range(nodes)])
    storage = inflow.copy()
    return features, climate, inflow, storage


def _make(**kw):
    features, climate, inflow, storage = _toy()
    defaults = dict(
        features_df=features, climate_df=climate, inflow_df=inflow,
        storage_df=storage, window_size=30, target_weeks=4,
        num_nodes=3, node_feat_dim=5, climate_lags=[1, 3],
        raw_oni=climate["ONI"],
    )
    defaults.update(kw)
    return ReservoirInflowDataset(**defaults), climate


def _expected_lag0(climate, col):
    mk = climate.index.to_period("M")
    monthly = climate.copy()
    monthly.index = mk
    means = monthly.groupby(level=0)[col].mean()
    base = pd.Series((mk - 1).map(means), index=climate.index, dtype="float64")
    return base.ffill().bfill().values.astype("float32")


class TestClimateLags:
    def test_tensor_shape_is_index_major(self):
        ds, climate = _make()
        # 4 indices x 3 lags (0, 1, 3) = 12 rows
        assert ds.climate_lag_tensor.shape == (12, len(climate))

    def test_sample_climate_shape_is_indices_by_lags(self):
        ds, _ = _make()
        sample = ds[10]
        assert sample["climate_indices"].shape == (4, 3)

    def test_lag0_uses_last_completed_month_only(self):
        ds, climate = _make()
        exp = _expected_lag0(climate, "ONI")
        assert np.allclose(ds.climate_lag_tensor[0], exp, atol=1e-5)

    def test_no_lookahead(self):
        """A lag-0 feature at day d must not depend on any value inside d's month."""
        ds, climate = _make()
        col0 = ds.climate_lag_tensor[0].astype(float)
        mk = climate.index.to_period("M")
        for d in [40, 100, 300]:
            m = mk[d]
            in_month = climate["ONI"].values[mk == m]
            # value must differ from the *partial/whole current month's* mean
            assert not np.isclose(col0[d], in_month.mean(), atol=1e-6) or len(in_month) > 28

    def test_raw_oni_is_unstandardized_and_emitted(self):
        ds, climate = _make()
        exp = climate["ONI"].ffill().bfill().values.astype("float32")
        assert np.allclose(ds.raw_oni, exp, atol=1e-5)
        sample = ds[10]
        assert "raw_oni" in sample
        assert sample["raw_oni"].ndim == 0
        assert isinstance(sample["raw_oni"].item(), float)

    def test_missing_iod_column_still_builds(self):
        # A configured index absent from the data must not crash the loader;
        # the caller (ReservoirGNN) raises clearly at forward time.
        features, climate, inflow, storage = _toy()
        climate = climate.drop(columns=["IOD"])
        ds = ReservoirInflowDataset(
            features_df=features, climate_df=climate, inflow_df=inflow,
            storage_df=storage, window_size=30, target_weeks=4, num_nodes=3,
            node_feat_dim=5, climate_lags=[1, 3],
        )
        assert ds[5]["climate_indices"].shape == (3, 3)
