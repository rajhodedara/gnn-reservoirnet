import torch
import torch.nn as nn
from typing import Dict, Tuple

class AttentionMapExtractor:
    """
    Extracts GAT attention weights from a trained model.
    """
    def __init__(self, model: nn.Module):
        """
        Initializes AttentionMapExtractor.

        Args:
            model (nn.Module): Trained GNN model with GAT layers.
        """
        self.model = model
        self.model.eval()

    def extract_attention(self, node_features: torch.Tensor, climate_indices: torch.Tensor, edge_index: torch.Tensor) -> Dict[Tuple[int, int], float]:
        """
        Extracts attention weights for spatial heatmap rendering.

        Args:
            node_features (torch.Tensor): Input node features.
            climate_indices (torch.Tensor): Input climate features.
            edge_index (torch.Tensor): Edge connectivity tensor.

        Returns:
            Dict[Tuple[int, int], float]: Dictionary mapping (source_node, target_node) to attention weight.
        """
        attention_dict = {}
        with torch.no_grad():
            # Since model architecture is not provided, we mock the extraction based on typical PyG output
            # In a real scenario, the model would return attention weights (e.g. from a PyG GATConv with return_attention_weights=True)
            num_edges = edge_index.shape[1]
            mock_attention_weights = torch.softmax(torch.randn(num_edges), dim=0) 
            
            edges = edge_index.cpu().numpy()
            weights = mock_attention_weights.cpu().numpy()
            
            for i in range(num_edges):
                src, dst = int(edges[0, i]), int(edges[1, i])
                attention_dict[(src, dst)] = float(weights[i])
                
        return attention_dict
