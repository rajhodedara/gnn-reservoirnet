import pytest
import torch
import torch.nn as nn
import numpy as np
import os
import json
from src.inference.predict import ReservoirPredictor
from src.explainability.attention_maps import AttentionMapExtractor
from src.explainability.integrated_gradients import IGExplainer

class MockModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Mocking a forward pass that outputs (batch, num_nodes, horizons, quantiles)
        
    def forward(self, node_features, climate_indices, edge_index):
        batch_size = node_features.size(0)
        # Mocking 3 nodes, 12 horizons, 3 quantiles, ensuring graph connection
        base = node_features.sum() + climate_indices.sum()
        return base * torch.ones(batch_size, 3, 12, 3)

def test_inference_predictor(tmp_path):
    model = MockModel()
    predictor = ReservoirPredictor(model=model, device='cpu')
    
    node_features = torch.randn(2, 10)
    climate_indices = torch.randn(2, 5)
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    
    preds = predictor.predict(node_features, climate_indices, edge_index)
    
    # Check shapes and types
    assert isinstance(preds, np.ndarray)
    assert preds.shape == (2, 3, 12, 3)
    
    # Check save functionality
    output_path = os.path.join(tmp_path, "preds.json")
    predictor.save_predictions(preds, output_path)
    assert os.path.exists(output_path)
    
    with open(output_path, 'r') as f:
        data = json.load(f)
        assert "predictions" in data
        assert len(data["predictions"]) == 2

def test_attention_maps():
    model = MockModel()
    extractor = AttentionMapExtractor(model)
    
    node_features = torch.randn(2, 10)
    climate_indices = torch.randn(2, 5)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    
    attn_dict = extractor.extract_attention(node_features, climate_indices, edge_index)
    
    # 3 edges in the mocked edge_index
    assert len(attn_dict) == 3
    for k, v in attn_dict.items():
        assert isinstance(v, float)
        assert 0.0 <= v <= 1.0

def test_integrated_gradients():
    # IG requires the wrapper to return a scalar or batched scalars.
    model = MockModel()
    edge_index = torch.zeros(2, 0, dtype=torch.long)
    explainer = IGExplainer(model, edge_index=edge_index, target_idx=0)
    
    node_features = torch.randn(2, 10, requires_grad=True)
    climate_indices = torch.randn(2, 5, requires_grad=True)
    
    # Call attribute
    attrs = explainer.attribute(node_features, climate_indices)
    
    assert "node_features" in attrs
    assert "climate_indices" in attrs
    
    assert attrs["node_features"].shape == (2, 10)
    assert attrs["climate_indices"].shape == (2, 5)
    
    # Test aggregation
    feature_groups = {
        "precipitation": [0, 1, 2],
        "soil_moisture": [3, 4]
    }
    agg = explainer.aggregate_by_group(attrs["node_features"], feature_groups)
    assert "precipitation" in agg
    assert "soil_moisture" in agg
    assert isinstance(agg["precipitation"], float)
