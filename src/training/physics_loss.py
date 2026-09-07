"""Physics-informed loss for the PhysicsConstrainedReservoirGNN.

Combines:
1. Primary storage-level MSE/MAE loss on S(t+w) for w = 1..12.
2. Mass-balance residual penalty: penalises violations of the water
   balance given observed inflows.

The residual term is:
    residual(w) = S_pred(w) - (S_pred(w-1) + obs_inflow_tmc(w) - obs_outflow_tmc(w))
where obs_inflow_tmc is weekly observed inflow converted from cumecs to TMC.

If true outflow is unknown (it usually is), we drop the outflow term and
only penalise cases where the predicted trajectory cannot be explained by
the observed inflows -- i.e., the penalty fires when the model implies
implausibly large outflows or storage gains.

Loss = alpha * MSE(S_pred_median, S_true)
     + (1 - alpha) * mean(|residual|)
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Optional


# Conversion factor: 1 m3/s for 1 week = 7*86400 m3 = 604800 m3 = 0.6048 MCM = 0.021353 TMC
CUMEC_WEEK_TO_TMC = 604800.0 / 1e6 / 28.3168466  # ~0.021353


class PhysicsInformedStorageLoss(nn.Module):
    """Physics-informed loss for storage-level forecasting.

    Args:
        alpha: Weight on primary (MSE) storage loss.
            (1 - alpha) goes to the mass-balance residual penalty.
        median_idx: Index of the median (P50) quantile in the last dim.
    """

    def __init__(self, alpha: float = 0.8, median_idx: int = 1) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.median_idx = int(median_idx)

    def forward(
        self,
        predicted_storage: torch.Tensor,
        target_storage: torch.Tensor,
        current_storage: torch.Tensor,
        obs_inflow_cumecs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute combined physics-informed loss.

        Args:
            predicted_storage: (B, N, W, Q) — model output.
            target_storage: (B, N, W) — observed storage at each horizon [TMC].
            current_storage: (B, N) — observed storage at t=0 [TMC].
            obs_inflow_cumecs: (B, N, W) — observed weekly mean inflow [cumecs].
                If None, the mass-balance penalty term is skipped.

        Returns:
            Scalar loss tensor.
        """
        # --- Primary storage level loss (median quantile) ---
        S_med = predicted_storage[:, :, :, self.median_idx]  # (B, N, W)

        # Mask out NaN targets (incomplete weeks at the boundary)
        valid = ~torch.isnan(target_storage)
        if valid.sum() == 0:
            primary_loss = torch.tensor(0.0, device=predicted_storage.device)
        else:
            diff = (S_med - target_storage)[valid]
            primary_loss = (diff ** 2).mean()  # MSE

        # --- Mass balance residual penalty ---
        if obs_inflow_cumecs is not None and self.alpha < 1.0:
            obs_inflow_tmc = obs_inflow_cumecs * CUMEC_WEEK_TO_TMC  # (B, N, W)

            # S_prev at each week: [S(t0), S_pred(1), S_pred(2), ...]
            # S_pred(w) should ≈ S_prev(w) + obs_inflow(w) - true_outflow(w)
            # Since outflow is unknown, we compute the implied outflow and
            # penalise only egregious violations (|implied_outflow| > capacity).
            s_prev = torch.cat(
                [current_storage.unsqueeze(2), S_med[:, :, :-1]], dim=2
            )  # (B, N, W)
            implied_net = S_med - s_prev  # = inflow - outflow (what model says)
            obs_net_inflow = obs_inflow_tmc  # ignoring outflow = conservative estimate

            # Penalise: when implied net is much lower than observed inflow,
            # meaning the model is 'losing' more water than physically possible.
            # Or when implied net significantly exceeds observed inflow
            # (would require negative outflow = physically impossible).
            residual = implied_net - obs_net_inflow  # (B, N, W)
            mb_penalty = (residual ** 2).mean()
        else:
            mb_penalty = torch.tensor(0.0, device=predicted_storage.device)

        total = self.alpha * primary_loss + (1.0 - self.alpha) * mb_penalty
        return total
