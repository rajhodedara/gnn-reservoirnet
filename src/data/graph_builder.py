import torch
from torch_geometric.data import Data
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any

def build_physical_edges(reservoirs_info: List[Dict[str, Any]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parse reservoir dicts, create directed edges following upstream->downstream river topology.
    Weight by inverse distance (or default to 1.0 if not provided).
    
    Args:
        reservoirs_info: List of dicts with 'id', 'name', 'upstream' (list of IDs).
        
    Returns:
        edge_index: Tensor of shape (2, num_edges)
        edge_weight: Tensor of shape (num_edges,)
    """
    id_to_idx = {res['id']: i for i, res in enumerate(reservoirs_info)}
    sources = []
    targets = []
    weights = []
    
    for res in reservoirs_info:
        current_idx = id_to_idx.get(res['id'])
        upstream_list = res.get('upstream', [])
        
        for up_id in upstream_list:
            if up_id in id_to_idx:
                up_idx = id_to_idx[up_id]
                sources.append(up_idx)
                targets.append(current_idx)
                
                # If distance info exists between these two, we could use it here.
                # For now, default weight is 1.0.
                dist = res.get('distance_to_downstream_km', 1.0)
                dist = max(dist, 1.0)
                weights.append(1.0 / dist)
            
    if not sources:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float)
        
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float)
    return edge_index, edge_weight

def get_region(lon: float, lat: float) -> str:
    """
    Determine if a reservoir is windward or leeward of the Western Ghats.
    Simplified rule: ~73.5E for northern, ~76E for southern.
    """
    ridge_lon = 76.0 - ((lat - 8.0) / 12.0) * 2.5
    if lon < ridge_lon:
        return 'windward'
    return 'leeward'

def build_climatological_edges(
    rainfall_data: pd.DataFrame, 
    reservoirs_info: List[Dict[str, Any]], 
    base_threshold: float = 0.6,
    ghats_threshold: float = 0.8
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute pairwise Pearson correlation of historical seasonal rainfall anomalies.
    Create undirected edges where correlation > threshold.
    
    Args:
        rainfall_data: DataFrame with reservoir IDs as columns and dates as index.
        reservoirs_info: List of dicts with 'id', 'lon', 'lat'.
        base_threshold: Threshold for correlation to create an edge.
        ghats_threshold: Threshold for correlation across Western Ghats.
        
    Returns:
        edge_index: Tensor of shape (2, num_edges)
        edge_weight: Tensor of shape (num_edges,)
    """
    ids = [res['id'] for res in reservoirs_info]
    id_to_idx = {res_id: i for i, res_id in enumerate(ids)}
    
    valid_cols = [col for col in ids if col in rainfall_data.columns]
    if not valid_cols:
         return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float)
         
    corr_matrix = rainfall_data[valid_cols].corr(method='pearson')
    
    sources = []
    targets = []
    weights = []
    
    res_dict = {res['id']: res for res in reservoirs_info}
    
    for i, id1 in enumerate(valid_cols):
        for j, id2 in enumerate(valid_cols):
            if i >= j:
                continue
            
            corr = corr_matrix.loc[id1, id2]
            if pd.isna(corr):
                continue
                
            res1 = res_dict[id1]
            res2 = res_dict[id2]
            
            reg1 = get_region(res1.get('longitude', res1.get('lon', 0.0)), res1.get('latitude', res1.get('lat', 0.0)))
            reg2 = get_region(res2.get('longitude', res2.get('lon', 0.0)), res2.get('latitude', res2.get('lat', 0.0)))
            
            threshold = ghats_threshold if reg1 != reg2 else base_threshold
            
            if corr > threshold:
                idx1 = id_to_idx[id1]
                idx2 = id_to_idx[id2]
                sources.extend([idx1, idx2])
                targets.extend([idx2, idx1])
                weights.extend([corr, corr])
                
    if not sources:
        return torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float)
        
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    edge_weight = torch.tensor(weights, dtype=torch.float)
    return edge_index, edge_weight

def build_reservoir_graph(
    reservoirs: List[Dict[str, Any]], 
    correlation_threshold: float = 0.6,
    block_cross_ghats: bool = True,
    use_physical: bool = True,
    use_climatological: bool = True,
    rainfall_data: pd.DataFrame = None
) -> Data:
    """
    Combines physical + climatological edges into a PyTorch Geometric Data object.
    
    Args:
        reservoirs: List of dicts with reservoir metadata.
        correlation_threshold: Threshold for climatological edge.
        block_cross_ghats: If True, blocks edges across Western Ghats unless threshold is higher.
        use_physical: Whether to include physical edges.
        use_climatological: Whether to include climatological edges.
        rainfall_data: Optional DataFrame with historical rainfall for correlations.
        
    Returns:
        Data object with edge_index, edge_type, and edge_weight.
    """
    if use_physical:
        phys_edge_index, phys_edge_weight = build_physical_edges(reservoirs)
    else:
        phys_edge_index, phys_edge_weight = torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float)
        
    if use_climatological and rainfall_data is not None:
        clim_edge_index, clim_edge_weight = build_climatological_edges(
            rainfall_data, reservoirs, 
            base_threshold=correlation_threshold, 
            ghats_threshold=1.0 if block_cross_ghats else correlation_threshold
        )
    else:
        clim_edge_index, clim_edge_weight = torch.empty((2, 0), dtype=torch.long), torch.empty((0,), dtype=torch.float)
    
    num_phys = phys_edge_index.size(1) if phys_edge_index.dim() == 2 else 0
    num_clim = clim_edge_index.size(1) if clim_edge_index.dim() == 2 else 0
    
    if num_phys == 0 and num_clim == 0:
        return Data(
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_weight=torch.empty((0,), dtype=torch.float),
            edge_type=torch.empty((0,), dtype=torch.long),
            num_nodes=len(reservoirs)
        )
        
    edge_index = torch.cat([phys_edge_index, clim_edge_index], dim=1) if num_phys > 0 and num_clim > 0 else (phys_edge_index if num_phys > 0 else clim_edge_index)
    edge_weight = torch.cat([phys_edge_weight, clim_edge_weight], dim=0) if num_phys > 0 and num_clim > 0 else (phys_edge_weight if num_phys > 0 else clim_edge_weight)
    
    phys_type = torch.zeros(num_phys, dtype=torch.long)
    clim_type = torch.ones(num_clim, dtype=torch.long)
    edge_type = torch.cat([phys_type, clim_type], dim=0) if num_phys > 0 and num_clim > 0 else (phys_type if num_phys > 0 else clim_type)
    
    return Data(
        edge_index=edge_index,
        edge_weight=edge_weight,
        edge_type=edge_type,
        num_nodes=len(reservoirs)
    )
