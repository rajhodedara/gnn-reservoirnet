import torch
import torch.nn as nn


class QuantilePredictionHead(nn.Module):
    """Direct Multi-Step Quantile Prediction Head.
    
    Predicts P10, P50, and P90 inflows directly for weeks +1 to +12.
    """
    
    def __init__(self, input_dim: int, num_weeks: int = 12, num_quantiles: int = 3, hidden_dim: int = 128) -> None:
        """Initializes the QuantilePredictionHead.
        
        Args:
            input_dim: Dimensionality of the fused spatiotemporal + climate embedding.
            num_weeks: Number of weeks to predict (default: 12).
            num_quantiles: Number of quantiles to predict (default: 3 for P10/P50/P90).
            hidden_dim: Hidden layer dimensionality.
        """
        super().__init__()
        self.num_weeks = num_weeks
        self.num_quantiles = num_quantiles
        
        # MLP for prediction
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_weeks * num_quantiles)
        )

    def forward(self, fused_embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            fused_embeddings (torch.Tensor): Fused features for each node.
                Shape: (batch_size, num_nodes, input_dim) or (batch_size * num_nodes, input_dim)
                
        Returns:
            torch.Tensor: Multi-step quantile predictions.
                Shape: (batch_size, num_nodes, num_weeks, num_quantiles) if input is 3D.
        """
        original_shape = fused_embeddings.shape
        is_3d = len(original_shape) == 3
        
        if is_3d:
            batch_size, num_nodes, input_dim = original_shape
            x = fused_embeddings.view(batch_size * num_nodes, input_dim)
        else:
            x = fused_embeddings
            
        out = self.mlp(x)
        
        if is_3d:
            out = out.view(batch_size, num_nodes, self.num_weeks, self.num_quantiles)
        else:
            out = out.view(-1, self.num_weeks, self.num_quantiles)
            
        return out
