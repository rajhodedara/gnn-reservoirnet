"""Regression tests for the weekly-sum un-scaling fix (main.py evaluate)."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.unscale import unscale_weekly_sum, scale_weekly_sum  # noqa: E402

DAYS = 7


def _make_standardized_daily(rng, batch=40, nodes=3, days=28):
    """Raw daily inflows + their per-node standardization, mirroring main.py."""
    raw = rng.gamma(shape=2.0, scale=50.0, size=(days, nodes))  # positive, skewed
    mean = raw.mean(axis=0)
    std = raw.std(axis=0).replace(0, 1) if hasattr(raw.std(axis=0), "replace") else raw.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    z = (raw - mean) / std
    # Weekly sums of z-scores, exactly as ReservoirInflowDataset.__getitem__
    n_weeks = days // DAYS
    z_weekly = z[: n_weeks * DAYS].reshape(n_weeks, DAYS, nodes).sum(axis=1)
    raw_weekly = raw[: n_weeks * DAYS].reshape(n_weeks, DAYS, nodes).sum(axis=1)
    # Tile to a fake batch dimension order (batch, nodes)
    z_batch = np.repeat(z_weekly[None, ...], batch, axis=0)
    raw_batch = np.repeat(raw_weekly[None, ...], batch, axis=0)
    return raw, mean, std, z, z_batch, raw_batch


class TestUnscaleWeeklySum:
    def test_exact_inverse_of_standardization(self, rng=None):
        rng = np.random.default_rng(7)
        raw, mean, std, z, z_batch, raw_batch = _make_standardized_daily(rng)
        n_weeks = raw.shape[0] // DAYS
        # Un-scale batch of weekly z-sums -> must equal raw weekly sums exactly
        recon = unscale_weekly_sum(z_batch, mean, std)
        assert recon.shape == z_batch.shape
        assert np.allclose(recon, np.repeat(raw_batch, 1, axis=0), atol=1e-8)

    def test_old_formula_bias_is_six_mu(self):
        """The historical bug (z*std + 1*mean) under-counts by exactly 6*mu."""
        rng = np.random.default_rng(11)
        raw = rng.uniform(10, 100, size=(28, 4))
        mean = raw.mean(axis=0)
        std = raw.std(axis=0)
        z = (raw - mean) / std
        z_weekly = z.reshape(4, DAYS, 4).sum(axis=1)
        truth = raw.reshape(4, DAYS, 4).sum(axis=1)
        old = z_weekly * std + mean          # buggy formula
        new = unscale_weekly_sum(z_weekly, mean, std)
        assert np.allclose(truth - old, (DAYS - 1) * mean, atol=1e-8)
        assert np.allclose(new, truth, atol=1e-8)

    def test_broadcast_over_quantile_axis(self):
        rng = np.random.default_rng(3)
        z = rng.normal(size=(5, 3, 3))  # (batch, nodes, quantiles)
        mean = np.array([10.0, 20.0, 30.0])
        std = np.array([2.0, 4.0, 5.0])
        out = unscale_weekly_sum(z, mean[:, None], std[:, None])
        assert out.shape == z.shape
        assert np.allclose(out[:, 0, 0], z[:, 0, 0] * 2.0 + DAYS * 10.0)

    def test_roundtrip(self):
        rng = np.random.default_rng(5)
        raw_sum = rng.uniform(100, 1000, size=(6, 3))
        mean = np.array([5.0, 8.0, 12.0])
        std = np.array([3.0, 6.0, 9.0])
        z = scale_weekly_sum(raw_sum, mean, std)
        assert np.allclose(unscale_weekly_sum(z, mean, std), raw_sum, atol=1e-9)
