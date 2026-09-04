import torch
import torch.nn as nn
from typing import Dict, Any
import numpy as np
import os
import json

class ReservoirPredictor:
    """
    Inference wrapper for the Reservoir GNN.
    Optimized for CPU inference environments (e.g., standard laptops).
    """
    def __init__(self, model: nn.Module, checkpoint_path: str = None, device: str = 'cpu'):
        """
        Initializes the predictor.

        Args:
            model (nn.Module): The untrained model architecture.
            checkpoint_path (str, optional): Path to the trained .pt weights.
            device (str): Device to run inference on. Defaults to 'cpu' for office laptops.
        """
        self.device = torch.device(device)
        self.model = model.to(self.device)
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading weights from {checkpoint_path} to {self.device}...")
            # Use map_location to ensure it loads on CPU even if trained on GPU
            state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
        
        self.model.eval()

    def predict(self, node_features: torch.Tensor, climate_indices: torch.Tensor, edge_index: torch.Tensor) -> np.ndarray:
        """
        Runs a forward pass to predict reservoir inflow quantiles.

        Args:
            node_features (torch.Tensor): Batched node features.
            climate_indices (torch.Tensor): Batched climate features (ENSO, IOD).
            edge_index (torch.Tensor): Graph topology.

        Returns:
            np.ndarray: Predictions of shape (batch, nodes, horizons, quantiles).
        """
        node_features = node_features.to(self.device)
        climate_indices = climate_indices.to(self.device)
        edge_index = edge_index.to(self.device)

        with torch.no_grad():
            # CPU inference does not need autocast typically, keeping it standard FP32
            preds = self.model(node_features, climate_indices, edge_index)
            
        return preds.cpu().numpy()
        
    def save_predictions(self, preds: np.ndarray, output_path: str):
        """
        Saves predictions to a JSON file for downstream visualization or dashboarding.
        """
        # Convert to list for JSON serialization
        # Format: [batch_idx][node_idx][horizon_idx] = [p10, p50, p90]
        preds_list = preds.tolist()
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump({"predictions": preds_list}, f)
        print(f"Predictions saved to {output_path}")

