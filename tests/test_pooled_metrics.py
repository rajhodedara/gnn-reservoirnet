"""Regression tests for pooled evaluation metrics.

Pooling must concatenate every reservoir's obs/pred BEFORE scoring; averaging
per-reservoir NSE across dams of different variance is not a valid summary
(that bug produced the misleading El Nino table).
"""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import Evaluator  # noqa: E402
from src.evaluation.metrics import nse  # noqa: E402


NAMES = ["A", "B", "C"]
BASINS = {"A": "x", "B": "x", "C": "y"}


def _synthetic(seed=0):
    rng = np.random.default_rng(seed)
    n = 400
    obs = np.stack([rng.gamma(2.0, 100.0, n) for _ in range(3)], axis=1)
    pred = obs + rng.normal(0, 40, obs.shape)
    ens = np.stack([pred * 0.9, pred, pred * 1.1], axis=-1)
    oni = rng.normal(0, 1.2, n)
    return obs, pred, ens, oni


def test_pooled_row_present_and_finite():
    obs, pred, ens, oni = _synthetic()
    ev = Evaluator(NAMES, BASINS)
    r = ev.evaluate(obs, pred, ens, oni)
    assert "pooled" in r
    p = r["pooled"]
    assert len(p) == 1
    assert np.isfinite(float(p.iloc[0]["NSE"]))
    assert int(p.iloc[0]["n_values"]) == obs.size
    assert int(p.iloc[0]["n_reservoirs"]) == 3


def test_pooled_equals_direct_computation():
    obs, pred, ens, oni = _synthetic(1)
    ev = Evaluator(NAMES, BASINS)
    r = ev.evaluate(obs, pred, ens, oni)
    expected = nse(obs.reshape(-1), pred.reshape(-1))
    got = float(r["pooled"].iloc[0]["NSE"])
    assert np.isclose(got, expected, equal_nan=True)


def test_pooled_differs_from_mean_of_per_reservoir():
    """Construct a case where they provably differ (the whole point)."""
    rng = np.random.default_rng(7)
    n = 300
    # Reservoir A: huge variance; B: tiny variance. Pooled is dominated by A.
    a = rng.normal(10000, 3000, n)
    b = rng.normal(1, 0.01, n)
    obs = np.stack([a, b], axis=1)
    pred = obs * 1.0
    pred[:, 0] = pred[:, 0] * 1.15          # big absolute error on A
    pred[:, 1] = pred[:, 1] * 1.0           # held accurate on B
    ens = np.stack([pred, pred, pred], axis=-1)
    oni = np.zeros(n)
    ev = Evaluator(["A", "B"], {"A": "x", "B": "x"})
    r = ev.evaluate(obs, pred, ens, oni)
    pooled = float(r["pooled"].iloc[0]["NSE"])
    mean_of = float(r["per_reservoir"]["NSE"].mean())
    assert not np.isclose(pooled, mean_of, atol=1e-6)


def test_enso_pooled_present():
    obs, pred, ens, oni = _synthetic(2)
    ev = Evaluator(NAMES, BASINS)
    r = ev.evaluate(obs, pred, ens, oni)
    assert "enso_comparison_pooled" in r
    ep = r["enso_comparison_pooled"]
    if not ep.empty:
        assert set(ep["Condition"]) <= {"El Nino", "Neutral"}
        assert "n_origins" in ep.columns
