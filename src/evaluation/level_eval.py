"""NumPy mass-balance recursion for reservoir storage (torch-free evaluation path)."""

import numpy as np


def mass_balance_numpy(
    s0_mcm: float,
    inflow_mcm_per_week: np.ndarray,
    releases_mcm_per_week: np.ndarray,
    cap_mcm: float,
) -> np.ndarray:
    """Sequential water balance: S(w) = clip(S(w-1) + I(w) - R(w), 0, cap).

    Args:
        s0_mcm: initial storage (MCM).
        inflow_mcm_per_week: weekly inflow volumes (MCM), length = horizon.
        releases_mcm_per_week: weekly release volumes (MCM), same length.
        cap_mcm: gross capacity (MCM).

    Returns:
        np.ndarray of predicted storage (MCM), length = horizon.
    """
    s = float(s0_mcm)
    out = []
    for i_vol, r_vol in zip(np.asarray(inflow_mcm_per_week, float), np.asarray(releases_mcm_per_week, float)):
        s = float(np.clip(s + i_vol - r_vol, 0.0, cap_mcm))
        out.append(s)
    return np.array(out)
