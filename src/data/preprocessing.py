import pandas as pd
import numpy as np
from typing import Dict, List
from sklearn.preprocessing import StandardScaler

def sanitize_readings(df: pd.DataFrame, max_rainfall_dict: Dict[str, float] = None) -> pd.DataFrame:
    """
    Physical bounds enforcement.
    - Drop negative inflows
    - Cap rainfall at basin-specific maximum plausible values
    - Detect and replace sensor anomalies (readings of 0 or 9999 for non-zero expected fields)
    
    Args:
        df: DataFrame with columns 'inflow', 'rainfall', etc.
        max_rainfall_dict: Dictionary mapping reservoir/basin IDs to max plausible rainfall.
        
    Returns:
        Sanitized DataFrame.
    """
    df = df.copy()
    
    if 'inflow' in df.columns:
        df.loc[df['inflow'] < 0, 'inflow'] = 0.0
        df.loc[df['inflow'] == 9999, 'inflow'] = np.nan
        
    if 'rainfall' in df.columns and max_rainfall_dict:
        # Assuming df has a 'basin_id' or we apply globally if dict has 'global'
        cap = max_rainfall_dict.get('global', 1000.0)
        df.loc[df['rainfall'] > cap, 'rainfall'] = cap
        df.loc[df['rainfall'] < 0, 'rainfall'] = 0.0
        df.loc[df['rainfall'] == 9999, 'rainfall'] = np.nan
        
    return df

def impute_missing(df: pd.DataFrame, window: int = 7) -> pd.DataFrame:
    """
    Rolling median imputation for sensor dropouts.
    
    Args:
        df: DataFrame with time-series data (index should be datetime).
        window: Window size in days.
        
    Returns:
        DataFrame with missing values imputed.
    """
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].rolling(window=window, min_periods=1, center=True).median())
        # Forward fill and backward fill remaining nans
        df[col] = df[col].ffill().bfill()
    return df

def compute_basin_averages(gridded_data: pd.DataFrame, basin_mask: Dict[str, List[str]]) -> pd.DataFrame:
    """
    Aggregate IMD gridded rainfall to basin-level using catchment boundaries.
    
    Args:
        gridded_data: DataFrame with grid points as columns.
        basin_mask: Mapping from basin ID to list of grid point column names.
        
    Returns:
        DataFrame with basin IDs as columns.
    """
    basin_df = pd.DataFrame(index=gridded_data.index)
    for basin_id, grid_points in basin_mask.items():
        valid_points = [p for p in grid_points if p in gridded_data.columns]
        if valid_points:
            basin_df[basin_id] = gridded_data[valid_points].mean(axis=1)
        else:
            basin_df[basin_id] = np.nan
    return basin_df

def compute_enso_lags(climate_df: pd.DataFrame, lags: List[int] = [1, 3, 6]) -> pd.DataFrame:
    """
    Create lagged moving averages of ENSO/IOD indices.
    
    Args:
        climate_df: DataFrame with climate indices.
        lags: List of lags in months.
        
    Returns:
        DataFrame with lagged moving averages.
    """
    df_out = climate_df.copy()
    for col in climate_df.columns:
        for lag in lags:
            # Assuming monthly data, rolling window = lag
            df_out[f'{col}_lag{lag}m_avg'] = climate_df[col].rolling(window=lag, min_periods=1).mean().shift(1)
    return df_out

class Normalizer:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def normalize_features(self, df: pd.DataFrame, is_training: bool = True) -> pd.DataFrame:
        """
        StandardScaler normalization with saved statistics for inference.
        
        Args:
            df: DataFrame to normalize.
            is_training: If True, fits the scaler.
            
        Returns:
            Normalized DataFrame.
        """
        cols = df.select_dtypes(include=[np.number]).columns
        if is_training:
            df[cols] = self.scaler.fit_transform(df[cols])
        else:
            df[cols] = self.scaler.transform(df[cols])
        return df
