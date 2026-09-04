import pytest
import torch
import pandas as pd
import numpy as np
from src.data.graph_builder import build_physical_edges, get_region, build_climatological_edges

# Mock reservoir data based on the YAML structure
MOCK_RESERVOIRS = [
    {"id": "resA", "upstream": []},
    {"id": "resB", "upstream": ["resA"]},
    {"id": "resC", "upstream": ["resB"]},
    {"id": "resD", "upstream": ["resA", "resC"]},
]

def test_build_physical_edges_direction():
    """Test that physical edges are built upstream -> downstream."""
    edge_index, edge_weight = build_physical_edges(MOCK_RESERVOIRS)
    
    assert edge_index.shape[0] == 2
    assert edge_index.shape[1] == 4  # A->B, B->C, A->D, C->D
    
    sources = edge_index[0].tolist()
    targets = edge_index[1].tolist()
    
    id_to_idx = {res['id']: i for i, res in enumerate(MOCK_RESERVOIRS)}
    
    assert sources[0] == id_to_idx["resA"]
    assert targets[0] == id_to_idx["resB"]
    
    assert sources[1] == id_to_idx["resB"]
    assert targets[1] == id_to_idx["resC"]
    
    assert sources[2] == id_to_idx["resA"]
    assert targets[2] == id_to_idx["resD"]
    
    assert sources[3] == id_to_idx["resC"]
    assert targets[3] == id_to_idx["resD"]

def test_build_physical_edges_empty():
    """Test with no physical connections."""
    reservoirs = [{"id": "resA", "upstream": []}, {"id": "resB", "upstream": []}]
    edge_index, edge_weight = build_physical_edges(reservoirs)
    
    assert edge_index.shape == (2, 0)
    assert edge_weight.shape == (0,)

def test_get_region():
    """Test windward/leeward classification."""
    # Somewhere west of Western Ghats (e.g. Mangalore)
    assert get_region(74.0, 12.0) == 'windward'
    
    # Somewhere east of Western Ghats (e.g. Bangalore/Hyderabad)
    assert get_region(77.0, 12.0) == 'leeward'

def test_build_climatological_edges():
    """Test that climatological edges connect highly correlated nodes within regions."""
    reservoirs = [
        {"id": "windward1", "lat": 13.0, "lon": 74.0},
        {"id": "windward2", "lat": 14.0, "lon": 74.5},
        {"id": "leeward1", "lat": 13.0, "lon": 77.0},
        {"id": "leeward2", "lat": 14.0, "lon": 77.5},
    ]
    
    # Create perfectly correlated data for same regions, and anti-correlated across
    # Actually, let's just make windward1/2 correlated (1.0), leeward1/2 correlated (1.0)
    # and cross-region correlated at 0.7.
    # We'll set base_threshold=0.6, ghats_threshold=0.8
    # Since cross-region correlation (0.7) < ghats_threshold (0.8), no cross edge.
    # Since same-region correlation (1.0) > base_threshold (0.6), yes edge.
    
    dates = pd.date_range('2020-01-01', periods=5)
    data = {
        "windward1": [1, 2, 3, 4, 5],
        "windward2": [1, 2, 3, 4, 5],
        "leeward1": [5, 4, 3, 2, 1],
        "leeward2": [5, 4, 3, 2, 1],
    }
    df = pd.DataFrame(data, index=dates)
    
    edge_index, edge_weight = build_climatological_edges(
        rainfall_data=df,
        reservoirs_info=reservoirs,
        base_threshold=0.6,
        ghats_threshold=0.8
    )
    
    id_to_idx = {r['id']: i for i, r in enumerate(reservoirs)}
    
    # Should have edges: W1-W2, W2-W1, L1-L2, L2-L1
    assert edge_index.shape[1] == 4
    
    edges_set = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    expected = {
        (id_to_idx["windward1"], id_to_idx["windward2"]),
        (id_to_idx["windward2"], id_to_idx["windward1"]),
        (id_to_idx["leeward1"], id_to_idx["leeward2"]),
        (id_to_idx["leeward2"], id_to_idx["leeward1"]),
    }
    
    assert edges_set == expected
