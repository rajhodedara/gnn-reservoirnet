"""Torch-free tests for the Stage-2 level forecast logic (mass-balance recursion)."""

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.level_eval import mass_balance_numpy  # noqa: E402


class TestMassBalanceNumpy:
    def test_constant_release_drains_linearly(self):
        s0 = 100.0
        i = np.full(4, 5.0)    # inflow 5 MCM/week
        r = np.full(4, 10.0)   # releases 10 MCM/week
        cap = 200.0
        traj = mass_balance_numpy(s0, i, r, cap)
        assert traj.shape == (4,)
        assert traj[0] == pytest.approx(95.0)
        assert traj[3] == pytest.approx(80.0)

    def test_clipped_at_zero_and_cap(self):
        i = np.full(3, 1.0)
        r = np.full(3, 500.0)  # massive releases
        traj = mass_balance_numpy(50.0, i, r, 100.0)
        assert (traj >= 0).all()

    def test_gain_clipped_at_cap(self):
        i = np.full(3, 90.0)
        r = np.full(3, 1.0)
        traj = mass_balance_numpy(95.0, i, r, 100.0)
        assert (traj <= 100.0).all()
