import pytest
import torch
from src.models.gat_spatial import SpatialGAT
from src.models.tcn_temporal import TemporalTCN
from src.models.climate_attention import ClimateCrossAttention
from src.models.reservoir_gnn import ReservoirGNN

def test_spatial_gat():
    batch_size = 2
    num_nodes = 5
    in_channels = 16
    out_channels = 32
    
    model = SpatialGAT(in_channels=in_channels, hidden_channels=24, out_channels=out_channels, num_heads=2)
    
    # Simulate a batched graph
    x = torch.randn(batch_size * num_nodes, in_channels)
    
    # Fully connected graph for 5 nodes
    edge_index = torch.tensor([
        [0, 1, 2, 3, 4, 1, 2],
        [1, 0, 3, 2, 0, 4, 1]
    ], dtype=torch.long)
    
    # Duplicate edges for batching (simplified)
    # PyG DataLoader does this automatically, we just test if the forward pass runs
    
    out = model(x, edge_index)
    assert out.shape == (batch_size * num_nodes, out_channels)

def test_temporal_tcn():
    batch_size = 2
    num_nodes = 5
    tcn_in = 10
    lookback = 90
    channels = [16, 32]
    
    model = TemporalTCN(num_inputs=tcn_in, num_channels=channels)
    
    # Input shape: (B, C, L)
    x = torch.randn(batch_size * num_nodes, tcn_in, lookback)
    
    out = model(x)
    assert out.shape == (batch_size * num_nodes, channels[-1])

def test_climate_attention():
    batch_size = 2
    input_dim = 6
    embed_dim = 16
    
    model = ClimateCrossAttention(input_dim=input_dim, embed_dim=embed_dim, num_heads=2)
    
    # e.g., 3 ENSO indices, 1 IOD index, with lag_window=6
    enso = torch.randn(batch_size, 3, input_dim)
    iod = torch.randn(batch_size, 1, input_dim)
    
    out = model(enso, iod)
    assert out.shape == (batch_size, embed_dim)

def test_reservoir_gnn_forward():
    batch_size = 4
    num_nodes = 10
    spatial_in = 6
    lookback = 90
    num_weeks = 12
    num_quantiles = 3
    
    config = {
        "spatial_in_channels": spatial_in,
        "spatial_hidden": 16,
        "spatial_out": 16,
        "tcn_in_channels": spatial_in,
        "tcn_channels": [16, 16],
        "climate_input": lookback,
        "climate_embed": 16,
        "num_weeks": num_weeks,
        "num_quantiles": num_quantiles
    }
    
    model = ReservoirGNN(config)
    
    # Inputs matching the actual forward() signature
    node_features = torch.randn(batch_size, num_nodes, lookback, spatial_in)
    climate_indices = torch.randn(batch_size, 4, lookback)  # 4 climate indices
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    
    out = model(node_features, climate_indices, edge_index)
    
    assert out.shape == (batch_size, num_nodes, num_weeks, num_quantiles)

def test_mass_balance():
    batch_size = 2
    num_nodes = 3
    num_weeks = 4
    num_quantiles = 3
    
    config = {"num_weeks": num_weeks, "num_quantiles": num_quantiles}
    model = ReservoirGNN(config)
    
    predicted_inflows = torch.ones(batch_size, num_nodes, num_weeks, num_quantiles) * 100.0
    current_storage = torch.ones(batch_size, num_nodes, 1) * 500.0
    
    t_mean = torch.ones(batch_size, num_nodes, num_weeks) * 30.0
    t_max = torch.ones(batch_size, num_nodes, num_weeks) * 35.0
    t_min = torch.ones(batch_size, num_nodes, num_weeks) * 25.0
    ra = torch.ones(batch_size, num_nodes, num_weeks) * 15.0
    surface_area = torch.ones(batch_size, num_nodes, num_weeks) * 10.0
    releases = torch.ones(batch_size, num_nodes, num_weeks) * 50.0
    
    predicted_storage = model.predict_storage(
        predicted_inflows, current_storage, t_mean, t_max, t_min, ra, surface_area, releases
    )
    
    assert predicted_storage.shape == (batch_size, num_nodes, num_weeks, num_quantiles)
    # 500 + 100 - (some evap) - 50 = ~550 - evap
    # Evap rate = 0.0023 * (30 + 17.8) * sqrt(10) * 15 = 0.0023 * 47.8 * 3.16 * 15 = ~5.2
    # Evap vol = 5.2 * 10 = ~52
    # Next storage = 500 + 100 - 52 - 50 = 498
    assert predicted_storage[0, 0, 0, 0] > 400.0
