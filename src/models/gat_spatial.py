import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv
from typing import Tuple


class SpatialGAT(nn.Module):
    """Graph Attention Network spatial module for reservoir prediction.
    
    Implements multi-head GAT layers over heterogeneous edges (physical and climatological).
    Uses PyTorch Geometric's GATv2Conv.
    
    Attributes:
        in_channels (int): Dimensionality of input node features.
        hidden_channels (int): Dimensionality of hidden node features.
        out_channels (int): Dimensionality of output node embeddings.
        num_heads (int): Number of attention heads.
        dropout (float): Dropout probability.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        """Initializes the SpatialGAT module.
        
        Args:
            in_channels: Input feature dimension.
            hidden_channels: Hidden feature dimension.
            out_channels: Output embedding dimension.
            num_heads: Number of attention heads for GATv2Conv.
            dropout: Dropout probability.
        """
        super().__init__()
        
        # First GATv2 layer
        self.gat1 = GATv2Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=num_heads,
            concat=True,
            dropout=dropout,
            add_self_loops=True,
        )
        
        # Second GATv2 layer
        self.gat2 = GATv2Conv(
            in_channels=hidden_channels * num_heads,
            out_channels=out_channels,
            heads=1,  # Output layer, single head or average
            concat=False,
            dropout=dropout,
            add_self_loops=True,
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Forward pass of the SpatialGAT module.
        
        Args:
            x (torch.Tensor): Node features. Shape: (batch_size, num_nodes, feature_dim) or (num_nodes, feature_dim).
                If batched, assumed to be processed efficiently or flattened if needed. PyG usually expects (N, F).
                To handle batching properly with standard PyG, typically graphs are batched into a large disconnected graph.
            edge_index (torch.Tensor): Edge indices. Shape: (2, num_edges).
            edge_attr (torch.Tensor, optional): Edge attributes/types. Shape: (num_edges, edge_dim).
            
        Returns:
            torch.Tensor: Spatial embeddings. Shape matching input batch/nodes with out_channels.
        """
        # Note: If x is (batch_size, num_nodes, feature_dim), it should be reshaped 
        # for standard PyG processing or handled via a batched Data object.
        # Assuming x is (N_total, F) where N_total = batch_size * num_nodes, 
        # or x is processed per graph.
        
        # GAT layer 1
        x = self.gat1(x, edge_index, edge_attr=edge_attr)
        x = F.elu(x)
        x = self.dropout(x)
        
        # GAT layer 2
        x = self.gat2(x, edge_index, edge_attr=edge_attr)
        
        return x
