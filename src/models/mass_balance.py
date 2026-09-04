import torch
import torch.nn as nn


class MassBalanceLayer(nn.Module):
    """Deterministic (non-trainable) Mass-Balance Layer.
    
    Computes storage based on: S(t+1) = S(t) + I(t) - E(t) - R(t)
    Evaporation E(t) is computed using the Hargreaves-Samani equation.
    Releases R(t) are expected to be provided based on rule curves/tables.
    """
    
    def __init__(self) -> None:
        """Initializes the MassBalanceLayer (no trainable parameters)."""
        super().__init__()
        
    def hargreaves_samani_evaporation(
        self, t_mean: torch.Tensor, t_max: torch.Tensor, t_min: torch.Tensor, ra: torch.Tensor
    ) -> torch.Tensor:
        """Computes evaporation (mm/day or standard unit) using Hargreaves-Samani.
        
        E = 0.0023 * (T_mean + 17.8) * (T_max - T_min)^0.5 * Ra
        
        Args:
            t_mean: Mean temperature.
            t_max: Maximum temperature.
            t_min: Minimum temperature.
            ra: Extraterrestrial radiation.
            
        Returns:
            torch.Tensor: Evaporation rate.
        """
        # Clamp delta T to prevent negative values before square root
        delta_t = torch.clamp(t_max - t_min, min=0.0)
        e = 0.0023 * (t_mean + 17.8) * torch.sqrt(delta_t) * ra
        return torch.clamp(e, min=0.0)
        
    def forward(
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
        """Applies the mass balance equation over the prediction horizon.
        
        Args:
            predicted_inflows (torch.Tensor): P10/P50/P90 inflows. Shape: (batch, nodes, weeks, quantiles)
            current_storage (torch.Tensor): Initial storage S(t=0). Shape: (batch, nodes, 1, 1) or broadcastable
            t_mean (torch.Tensor): Temp forecasts. Shape: (batch, nodes, weeks)
            t_max (torch.Tensor): Temp forecasts. Shape: (batch, nodes, weeks)
            t_min (torch.Tensor): Temp forecasts. Shape: (batch, nodes, weeks)
            ra (torch.Tensor): Radiation. Shape: (batch, nodes, weeks)
            surface_area (torch.Tensor): Estimated surface area. Shape: (batch, nodes, weeks)
            releases (torch.Tensor): Releases from rule curves. Shape: (batch, nodes, weeks)
            
        Returns:
            torch.Tensor: Predicted storage. Shape: (batch, nodes, weeks, quantiles)
        """
        batch_size, num_nodes, num_weeks, num_quantiles = predicted_inflows.shape
        
        # Compute evaporation rate (per unit area)
        e_rate = self.hargreaves_samani_evaporation(t_mean, t_max, t_min, ra) # (batch, nodes, weeks)
        
        # Convert evaporation rate to volume using surface area (requires unit conversion in practice)
        # Assuming e_rate * surface_area gives volume in same units as inflows/storage (e.g., MCM)
        e_volume = e_rate * surface_area # (batch, nodes, weeks)
        
        # Expand e_volume and releases to match quantiles
        e_volume_q = e_volume.unsqueeze(-1).expand(-1, -1, -1, num_quantiles)
        releases_q = releases.unsqueeze(-1).expand(-1, -1, -1, num_quantiles)
        
        predicted_storage = torch.zeros_like(predicted_inflows)
        
        # Sequential mass balance
        s_t = current_storage.expand(-1, -1, num_quantiles) if current_storage.dim() == 3 else current_storage.squeeze(-2).expand(-1, -1, num_quantiles)
        
        for w in range(num_weeks):
            i_t = predicted_inflows[:, :, w, :]
            e_t = e_volume_q[:, :, w, :]
            r_t = releases_q[:, :, w, :]
            
            s_next = s_t + i_t - e_t - r_t
            # Storage cannot be negative
            s_next = torch.clamp(s_next, min=0.0)
            
            predicted_storage[:, :, w, :] = s_next
            s_t = s_next
            
        return predicted_storage
