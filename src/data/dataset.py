import torch
from torch.utils.data import Dataset, Sampler
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Iterator, Optional

class ReservoirInflowDataset(Dataset):
    """
    Loads processed temporal data for all reservoirs and creates sliding windows of 90-day lookback.
    Targets are aggregated inflow volumes.
    """
    def __init__(
        self,
        features_df: pd.DataFrame,
        climate_df: pd.DataFrame,
        inflow_df: pd.DataFrame,
        storage_df: pd.DataFrame,
        window_size: int = 90,
        target_weeks: int = 12,
        num_nodes: int = 10,
        node_feat_dim: int = 6
    ):
        """
        Args:
            features_df: Daily node features (nodes * features columns).
            climate_df: Monthly/Daily climate indices aligned to features_df index.
            inflow_df: Daily inflows for targets.
            storage_df: Daily storage.
            window_size: Lookback window in days.
            target_weeks: Forecast horizon in weeks.
            num_nodes: Number of reservoirs in the graph.
            node_feat_dim: Number of features per reservoir.
        """
        self.window_size = window_size
        self.target_horizon = target_weeks * 7
        self.num_nodes = num_nodes
        self.node_feat_dim = node_feat_dim
        
        # Ensure alignment
        common_idx = features_df.index.intersection(climate_df.index).intersection(inflow_df.index).intersection(storage_df.index)
        
        self.features = torch.FloatTensor(features_df.loc[common_idx].values)
        self.climate = torch.FloatTensor(climate_df.loc[common_idx].values)
        self.inflow = torch.FloatTensor(inflow_df.loc[common_idx].values)
        self.storage = torch.FloatTensor(storage_df.loc[common_idx].values)
        self.dates = common_idx
        
        # We need an ONI column in climate_df for the sampler
        self.oni_idx = climate_df.columns.get_loc('ONI') if 'ONI' in climate_df.columns else 0
        
        self.valid_indices = self._get_valid_indices()
        
    def _get_valid_indices(self) -> List[int]:
        valid = []
        n_samples = len(self.features)
        for i in range(self.window_size, n_samples - self.target_horizon):
            valid.append(i)
        return valid

    def __len__(self) -> int:
        return len(self.valid_indices)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        end_idx = self.valid_indices[idx]
        start_idx = end_idx - self.window_size
        
        # Original shape: (window_size, num_nodes * node_feat_dim)
        raw_features = self.features[start_idx:end_idx]
        
        # Reshape to (window_size, num_nodes, node_feat_dim) and then (num_nodes, window_size, node_feat_dim)
        node_features = raw_features.view(self.window_size, self.num_nodes, self.node_feat_dim).permute(1, 0, 2)
        
        # Climate indices shape: (window_size, num_climate_indices) => typically we just use a lag if needed, or pass the sequence
        # The model expects climate to be (batch, num_climate, num_lags).
        # We will permute it to (num_climate, window_size) and let the model handle lags from the window
        climate_indices = self.climate[start_idx:end_idx].permute(1, 0)
        
        current_storage = self.storage[end_idx - 1]
        oni_value = self.climate[end_idx - 1, self.oni_idx]
        
        # Aggregated weekly target inflow (num_nodes, target_weeks)
        raw_target = self.inflow[end_idx:end_idx + self.target_horizon]
        target_inflow = raw_target.view(self.target_horizon // 7, 7, self.num_nodes).sum(dim=1).permute(1, 0)
        
        return {
            'node_features': node_features,
            'climate_indices': climate_indices,
            'targets': target_inflow,
            'current_storage': current_storage,
            'oni': oni_value
        }

def create_temporal_splits(dates: pd.DatetimeIndex, val_years: List[int], test_years: List[int]) -> Tuple[List[int], List[int], List[int]]:
    """
    Split data into train/val/test by year, ensuring El Nino validation years are held out.
    
    Args:
        dates: DatetimeIndex of the dataset.
        val_years: Years for validation.
        test_years: Years for testing.
        
    Returns:
        train_idx, val_idx, test_idx
    """
    years = dates.year
    train_idx = np.where(~years.isin(val_years + test_years))[0].tolist()
    val_idx = np.where(years.isin(val_years))[0].tolist()
    test_idx = np.where(years.isin(test_years))[0].tolist()
    return train_idx, val_idx, test_idx

class StratifiedENSOSampler(Sampler):
    """
    Custom sampler ensuring each batch contains samples from El Nino (ONI > 0.5), 
    La Nina (ONI < -0.5), and Neutral periods.
    """
    def __init__(self, dataset: ReservoirInflowDataset, batch_size: int, subset_indices: Optional[List[int]] = None):
        self.dataset = dataset
        self.batch_size = batch_size
        self.indices = subset_indices if subset_indices is not None else list(range(len(dataset)))
        
        self.el_nino = []
        self.la_nina = []
        self.neutral = []
        
        for idx in self.indices:
            real_idx = self.dataset.valid_indices[idx]
            oni = self.dataset.climate[real_idx - 1, self.dataset.oni_idx].item()
            if oni > 0.5:
                self.el_nino.append(idx)
            elif oni < -0.5:
                self.la_nina.append(idx)
            else:
                self.neutral.append(idx)
                
    def __iter__(self) -> Iterator[List[int]]:
        el_nino = np.random.permutation(self.el_nino).tolist()
        la_nina = np.random.permutation(self.la_nina).tolist()
        neutral = np.random.permutation(self.neutral).tolist()
        
        batches = []
        while el_nino and la_nina and neutral:
            batch = []
            # Take a balanced mix if possible
            n_each = self.batch_size // 3
            
            if len(el_nino) < n_each or len(la_nina) < n_each or len(neutral) < n_each:
                break
                
            batch.extend([el_nino.pop() for _ in range(n_each)])
            batch.extend([la_nina.pop() for _ in range(n_each)])
            batch.extend([neutral.pop() for _ in range(self.batch_size - 2 * n_each)])
            
            np.random.shuffle(batch)
            batches.append(batch)
            
        np.random.shuffle(batches)
        for batch in batches:
            yield batch
            
    def __len__(self) -> int:
        return min(len(self.el_nino), len(self.la_nina), len(self.neutral)) // (self.batch_size // 3)
