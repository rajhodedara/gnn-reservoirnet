"""Exact inverse of per-day standardization applied to multi-day inflow sums.

The dataset targets are sums of daily z-scores over ``days`` days:

    sum_z = (sum_i x_i - days * mu) / sigma

so the exact raw-sum inverse is:

    sum_x = sum_z * sigma + days * mu

The historical bug was ``sum_z * sigma + mu`` (one day's mean instead of
``days`` days), which biased every reported weekly volume by ``-(days-1) * mu``
per reservoir.
"""

import numpy as np


def unscale_weekly_sum(z, mean, std, days: int = 7):
    """Invert per-day standardization for a multi-day sum of z-scores.

    Args:
        z: array of weekly z-score sums. ``mean``/``std`` must broadcast
            against it (e.g. z shape (batch, nodes) with 1-D mean/std, or
            z shape (batch, nodes, quantiles) with mean/std shaped (nodes, 1)).
        mean: per-node daily mean (pre-standardization).
        std: per-node daily std (pre-standardization).
        days: number of days summed into each z value.

    Returns:
        np.ndarray of raw multi-day sums, same shape as ``z``.
    """
    z = np.asarray(z, dtype=float)
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    return z * std + days * mean


def scale_weekly_sum(raw_sum, mean, std, days: int = 7):
    """Forward direction of :func:`unscale_weekly_sum` (for tests/sanity)."""
    raw_sum = np.asarray(raw_sum, dtype=float)
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    return (raw_sum - days * mean) / std
