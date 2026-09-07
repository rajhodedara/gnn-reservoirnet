"""Physics-Constrained Reservoir GNN.

Wraps the existing sub-modules (SpatialGAT, TemporalTCN, ClimateCrossAttention)
with a net_change_head that predicts weekly delta-storage, then rolls the mass
balance forward autoregressively inside forward().

Mass balance equation applied per week:
    S(t+w) = clip(S(t+w-1) + delta_S_pred(w), 0, gross_cap_tmc)

Units: Storage in TMC (thousand million cubic feet).
       1 MCM = 1/28.3168466 TMC.

Design notes
------------
* All existing sub-modules are reused as-is (additive change only).
* net_change_head predicts weekly Δ-storage (can be negative = net outflow).
* Autoregressive rollout is fully differentiable (no detach()).
* gross_caps registered as a buffer to auto-move with .to(device).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, Any, Optional

from .gat_spatial import SpatialGAT
from .tcn_temporal import TemporalTCN
from .climate_attention import ClimateCrossAttention
from .quantile_head import QuantilePredictionHead


class PhysicsConstrainedReservoirGNN(nn.Module):
    """GNN with differentiable mass-balance rollout for storage-level forecasts.

    Args:
        config: Dict with same keys as ReservoirGNN plus:
            ``gross_caps_tmc`` (list[float] | None): one cap per reservoir node.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()

        spatial_in_channels = config.get("spatial_in_channels", 64)
        spatial_hidden = config.get("spatial_hidden", 128)
        spatial_out = config.get("spatial_out", 64)

        tcn_in_channels = config.get("tcn_in_channels", 32)
        tcn_channels = config.get("tcn_channels", [64, 64, 128])

        climate_input = config.get("climate_input", 6)
        climate_embed = config.get("climate_embed", 32)

        fused_dim = spatial_out + tcn_channels[-1] + climate_embed

        self.num_weeks: int = config.get("num_weeks", 12)
        self.num_quantiles: int = config.get("num_quantiles", 3)

        # Sub-modules -- identical init signature to ReservoirGNN
        self.spatial_module = SpatialGAT(
            in_channels=spatial_in_channels,
            hidden_channels=spatial_hidden,
            out_channels=spatial_out,
            num_heads=config.get("gat_heads", 4),
            dropout=config.get("spatial", {}).get("dropout", 0.3),
        )

        self.temporal_module = TemporalTCN(
            num_inputs=tcn_in_channels,
            num_channels=tcn_channels,
            kernel_size=config.get("tcn_kernel", 3),
            dropout=config.get("temporal", {}).get("dropout", 0.4),
        )

        self.climate_module = ClimateCrossAttention(
            input_dim=climate_input,
            embed_dim=climate_embed,
            num_heads=config.get("climate_heads", 4),
            dropout=config.get("climate", {}).get("dropout", 0.1),
        )

        # Net-change head: predicts weekly delta-S [TMC] per quantile.
        self.net_change_head = QuantilePredictionHead(
            input_dim=fused_dim,
            num_weeks=self.num_weeks,
            num_quantiles=self.num_quantiles,
            hidden_dim=config.get("head_hidden", 128),
        )

        # Per-node capacity buffer shape (1, N, 1, 1) -> broadcasts (B, N, W, Q)
        gross_caps = config.get("gross_caps_tmc", None)
        if gross_caps is not None:
            caps = torch.tensor(gross_caps, dtype=torch.float32).view(1, -1, 1, 1)
        else:
            caps = torch.tensor([float("inf")])
        self.register_buffer("gross_caps", caps)

    # ------------------------------------------------------------------

    def _fused_embedding(
        self,
        node_features: torch.Tensor,
        climate_indices: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return fused (B, N, fused_dim) embedding."""
        B, N, W, F = node_features.shape

        # Spatial
        spatial_x = node_features[:, :, -1, :].reshape(B * N, F)
        offset = torch.arange(0, B * N, N, device=edge_index.device).view(1, B, 1)
        batched_edge_index = edge_index.unsqueeze(1) + offset
        batched_edge_index = batched_edge_index.view(2, B * edge_index.size(1))
        batched_edge_attr = edge_attr.repeat(B, 1) if edge_attr is not None else None
        spatial_embeds = self.spatial_module(
            spatial_x, batched_edge_index, batched_edge_attr
        )  # (B*N, spatial_out)

        # Temporal
        temporal_x = node_features.permute(0, 1, 3, 2).reshape(B * N, F, W)
        temporal_embeds = self.temporal_module(temporal_x)  # (B*N, tcn_out)

        # Climate
        enso_indices = climate_indices[:, :3, :]
        iod_indices = climate_indices[:, -1:, :]
        climate_context = self.climate_module(enso_indices, iod_indices)  # (B, embed)

        # Fuse
        spatial_embeds = spatial_embeds.view(B, N, -1)
        temporal_embeds = temporal_embeds.view(B, N, -1)
        climate_context = climate_context.unsqueeze(1).expand(-1, N, -1)
        return torch.cat([spatial_embeds, temporal_embeds, climate_context], dim=-1)

    # ------------------------------------------------------------------

    def forward(
        self,
        node_features: torch.Tensor,
        climate_indices: torch.Tensor,
        edge_index: torch.Tensor,
        current_storage: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict storage trajectories via mass-balance rollout.

        Args:
            node_features: (B, N, lookback, F)
            climate_indices: (B, 4, lookback)
            edge_index: (2, E)
            current_storage: (B, N) storage at forecast origin [TMC].
            edge_attr: optional (E, edge_features).

        Returns:
            predicted_storage: (B, N, W, Q) — storage at end of each week [TMC].
        """
        fused = self._fused_embedding(node_features, climate_indices, edge_index, edge_attr)
        net_change = self.net_change_head(fused)  # (B, N, W, Q)

        Q = net_change.shape[3]
        s_t = current_storage.unsqueeze(-1).expand(-1, -1, Q)  # (B, N, Q)

        storage_steps = []
        for w in range(self.num_weeks):
            s_next = s_t + net_change[:, :, w, :]
            # Lower bound
            s_next = torch.clamp(s_next, min=0.0)
            # Upper bound per node
            if self.gross_caps.numel() > 1:
                cap = self.gross_caps[:, :, 0, 0]  # (1, N)
                s_next = torch.min(s_next, cap.unsqueeze(-1).expand_as(s_next))
            storage_steps.append(s_next.unsqueeze(2))
            s_t = s_next

        return torch.cat(storage_steps, dim=2)  # (B, N, W, Q)
