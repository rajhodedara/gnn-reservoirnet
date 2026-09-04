import torch
import torch.nn as nn
from typing import Dict, Any

from .gat_spatial import SpatialGAT
from .tcn_temporal import TemporalTCN
from .climate_attention import ClimateCrossAttention
from .quantile_head import QuantilePredictionHead
from .mass_balance import MassBalanceLayer


class ReservoirGNN(nn.Module):
    """Main model class that composes all GNN components for reservoir prediction.
    
    Orchestrates: climate_attention -> gat_spatial -> tcn_temporal -> fusion -> quantile_head.
    The mass_balance layer is used for post-prediction storage computation.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initializes the ReservoirGNN from a configuration dictionary.
        
        Args:
            config: Configuration dictionary specifying architectures and hyperparams.
        """
        super().__init__()
        
        # Configuration extraction
        spatial_in_channels = config.get("spatial_in_channels", 64)
        spatial_hidden = config.get("spatial_hidden", 128)
        spatial_out = config.get("spatial_out", 64)
        
        tcn_in_channels = config.get("tcn_in_channels", 32)
        tcn_channels = config.get("tcn_channels", [64, 64, 128])
        
        climate_input = config.get("climate_input", 6)
        climate_embed = config.get("climate_embed", 32)
        
        fused_dim = spatial_out + tcn_channels[-1] + climate_embed
        
        # Modules
        self.spatial_module = SpatialGAT(
            in_channels=spatial_in_channels,
            hidden_channels=spatial_hidden,
            out_channels=spatial_out,
            num_heads=config.get("gat_heads", 4),
            dropout=config.get("spatial", {}).get("dropout", 0.3)
        )
        
        self.temporal_module = TemporalTCN(
            num_inputs=tcn_in_channels,
            num_channels=tcn_channels,
            kernel_size=config.get("tcn_kernel", 3),
            dropout=config.get("temporal", {}).get("dropout", 0.4)
        )
        
        self.climate_module = ClimateCrossAttention(
            input_dim=climate_input,
            embed_dim=climate_embed,
            num_heads=config.get("climate_heads", 4),
            dropout=config.get("climate", {}).get("dropout", 0.1)
        )
        
        self.quantile_head = QuantilePredictionHead(
            input_dim=fused_dim,
            num_weeks=config.get("num_weeks", 12),
            num_quantiles=config.get("num_quantiles", 3),
            hidden_dim=config.get("head_hidden", 128)
        )
        
        self.mass_balance = MassBalanceLayer()
        
    def forward(
        self,
        node_features: torch.Tensor,
        climate_indices: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass to predict inflows.
        
        Args:
            node_features: Spatial and temporal features. Shape: (batch, num_nodes, window, features)
            climate_indices: Climate features. Shape: (batch, num_climate, window)
            edge_index: Graph connectivity. Shape: (2, num_edges)
            edge_attr: Edge types/features.
            
        Returns:
            torch.Tensor: Quantile inflow predictions. Shape: (batch, nodes, weeks, quantiles)
        """
        B, N, W, F = node_features.shape
        
        # 1. Spatial Embeddings
        # Extract last timestep for spatial features
        spatial_x = node_features[:, :, -1, :].reshape(B * N, F)
        
        # Batch edge_index
        offset = torch.arange(0, B * N, N, device=edge_index.device).view(1, B, 1)
        batched_edge_index = edge_index.unsqueeze(1) + offset # (2, B, E)
        batched_edge_index = batched_edge_index.view(2, B * edge_index.size(1))
        
        if edge_attr is not None:
            batched_edge_attr = edge_attr.repeat(B, 1)
        else:
            batched_edge_attr = None
            
        spatial_embeds = self.spatial_module(spatial_x, batched_edge_index, batched_edge_attr) # (B * N, spatial_out)
        
        # 2. Temporal Embeddings
        # Reshape to (B * N, F, W)
        temporal_x = node_features.permute(0, 1, 3, 2).reshape(B * N, F, W)
        temporal_embeds = self.temporal_module(temporal_x) # (B * N, tcn_out)
        
        # 3. Climate Context
        # Split climate indices
        enso_indices = climate_indices[:, :3, :] # (B, 3, W)
        iod_indices = climate_indices[:, -1:, :] # (B, 1, W)
        
        climate_context = self.climate_module(enso_indices, iod_indices) # (B, climate_embed)
        
        # 4. Fusion
        spatial_embeds = spatial_embeds.view(B, N, -1)
        temporal_embeds = temporal_embeds.view(B, N, -1)
        climate_context = climate_context.unsqueeze(1).expand(-1, N, -1)
        
        fused = torch.cat([spatial_embeds, temporal_embeds, climate_context], dim=-1) # (B, N, fused_dim)
        
        # 5. Direct Multi-Step Prediction
        inflows = self.quantile_head(fused)
        
        return inflows

    def predict_storage(
        self,
        predicted_inflows: torch.Tensor,
        current_storage: torch.Tensor,
        t_mean: torch.Tensor,
        t_max: torch.Tensor,
        t_min: torch.Tensor,
        ra: torch.Tensor,
        surface_area: torch.Tensor,
        releases: torch.Tensor
    ) -> torch.Tensor:
        """Applies deterministic mass-balance to predict storage from inflows.
        
        Args:
            predicted_inflows: Inflows from forward().
            current_storage: Initial storage.
            t_mean, t_max, t_min, ra: Weather variables for evaporation.
            surface_area: Dynamic surface area.
            releases: Prescribed releases.
            
        Returns:
            torch.Tensor: Predicted storage.
        """
        return self.mass_balance(
            predicted_inflows, current_storage, t_mean, t_max, t_min, ra, surface_area, releases
        )
