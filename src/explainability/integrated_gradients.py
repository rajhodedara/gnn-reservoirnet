import torch
from captum.attr import IntegratedGradients
from typing import Dict, List
import numpy as np

class IGExplainer:
    """
    Integrated Gradients explainability module.
    Attributes inflow predictions to input features.
    """
    def __init__(self, model: torch.nn.Module, edge_index: torch.Tensor, target_idx: int = 0):
        """
        Initializes IGExplainer.

        Args:
            model (torch.nn.Module): The trained PyTorch model.
            edge_index (torch.Tensor): Graph edge connectivity tensor.
            target_idx (int): Output index to attribute (e.g., forecasting horizon).
        """
        self.model = model
        self.model.eval()
        self.edge_index = edge_index
        self.target_idx = target_idx
        # Wrapping model to ensure it outputs a single scalar per sample for IG
        self.ig = IntegratedGradients(self._model_forward_wrapper)

    def _model_forward_wrapper(self, node_features: torch.Tensor, climate_indices: torch.Tensor) -> torch.Tensor:
        """
        Wrapper to extract specific prediction for IG.
        """
        # Assuming output is (batch, num_nodes, horizons, quantiles) or similar
        # We attribute for a specific node/horizon (can be customized)
        out = self.model(node_features, climate_indices, self.edge_index)
        # Assuming we want to attribute the median prediction (index 1 if 0.1, 0.5, 0.9)
        # of the first horizon (target_idx) and sum over nodes for global feature attribution
        return out[..., self.target_idx, 1].sum(dim=1)

    def attribute(self, node_features: torch.Tensor, climate_indices: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Compute integrated gradients attribution.

        Args:
            node_features (torch.Tensor): Input node features.
            climate_indices (torch.Tensor): Input climate features.

        Returns:
            Dict[str, np.ndarray]: Attributions for node and climate features.
        """
        node_baseline = torch.zeros_like(node_features)
        climate_baseline = torch.zeros_like(climate_indices)

        attributions = self.ig.attribute(
            inputs=(node_features, climate_indices),
            baselines=(node_baseline, climate_baseline),
            n_steps=50
        )

        return {
            'node_features': attributions[0].detach().cpu().numpy(),
            'climate_indices': attributions[1].detach().cpu().numpy()
        }
        
    def aggregate_by_group(self, attributions: np.ndarray, feature_groups: Dict[str, List[int]]) -> Dict[str, float]:
        """
        Aggregates attribution scores by feature groups.

        Args:
            attributions (np.ndarray): Attribution scores (e.g., node features).
            feature_groups (Dict[str, List[int]]): Dictionary mapping group name to feature indices.

        Returns:
            Dict[str, float]: Aggregated attribution score per group.
        """
        # Average across nodes and time steps, getting feature-wise importance
        feature_importance = np.mean(np.abs(attributions), axis=(0, 1, 2)) if attributions.ndim == 4 else np.mean(np.abs(attributions), axis=0)
        
        group_importance = {}
        for group, indices in feature_groups.items():
            group_importance[group] = float(np.sum([feature_importance[i] for i in indices]))
            
        return group_importance
