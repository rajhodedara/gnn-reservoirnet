import torch
import torch.nn as nn


class ClimateCrossAttention(nn.Module):
    """Cross-Attention Climate Interaction Module for ENSO and IOD modulation.
    
    Captures non-linear interactions between Pacific (ENSO) and Indian Ocean (IOD) states.
    
    Attributes:
        embed_dim (int): Dimensionality of the climate context vector.
        num_heads (int): Number of attention heads.
    """
    
    def __init__(self, input_dim: int, embed_dim: int, num_heads: int = 4, dropout: float = 0.1) -> None:
        """Initializes the ClimateCrossAttention module.
        
        Args:
            input_dim: Dimensionality of the input climate indices sequence (e.g. lag window size or feature projection).
            embed_dim: Output dimensionality for the climate context.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        
        # Projections to embedding dimension
        self.enso_proj = nn.Linear(input_dim, embed_dim)
        self.iod_proj = nn.Linear(input_dim, embed_dim)
        
        # Cross attention: ENSO attends to IOD, and vice-versa (or single direction depending on design)
        # Here we use ENSO as Query, IOD as Key/Value
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Final projection to get the context vector
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, enso_indices: torch.Tensor, iod_indices: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            enso_indices (torch.Tensor): ENSO features (ONI, SOI, Niño3.4). 
                Shape: (batch_size, num_enso_indices, lag_window)
            iod_indices (torch.Tensor): IOD features.
                Shape: (batch_size, num_iod_indices, lag_window)
                
        Returns:
            torch.Tensor: Global climate context vector. Shape: (batch_size, embed_dim)
        """
        # Flatten temporal/feature dimensions or treat num_indices as sequence
        # Assuming we project the lag_window to embed_dim, treating num_indices as sequence length
        
        # (batch_size, seq_len, embed_dim)
        enso_embed = self.enso_proj(enso_indices)  
        iod_embed = self.iod_proj(iod_indices)
        
        # Cross attention: Query=ENSO, Key=Value=IOD
        # (batch_size, num_enso_indices, embed_dim)
        attn_out, _ = self.cross_attn(query=enso_embed, key=iod_embed, value=iod_embed)
        
        # Aggregate across the sequence dimension (e.g., mean pooling)
        # (batch_size, embed_dim)
        context = attn_out.mean(dim=1)
        
        # FFN with residual and layer norm
        out = self.fc(context)
        out = self.layer_norm(out + context)
        
        return out
