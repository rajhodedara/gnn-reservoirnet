import pandas as pd
import numpy as np
import os
import glob
from typing import Dict, Optional
try:
    import xarray as xr
except ImportError:
    xr = None

def load_reservoir_data(filepath: str, reservoir_id: str) -> pd.DataFrame:
    """
    Load daily reservoir data (storage, inflow, outflow) from WRIS CSV/API format.
    Handles dates and missing values. Converts TMC to MCM if necessary.
    
    Args:
        filepath: Path to the CSV file.
        reservoir_id: ID of the reservoir.
        
    Returns:
        DataFrame indexed by date.
    """
    if not os.path.exists(filepath):
        return pd.DataFrame()
        
    df = pd.read_csv(filepath)
    
    # Handle common date formats
    date_col = next((c for c in df.columns if 'date' in c.lower()), None)
    if date_col:
        sample_series = df[date_col].dropna().astype(str)
        if not sample_series.empty:
            sample = sample_series.iloc[0].strip()
            if len(sample) >= 10 and sample[:4].isdigit() and sample[4] in ("-", "/"):
                df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
            elif len(sample) >= 10 and sample[:2].isdigit() and sample[2] in ("-", "/"):
                df['Date'] = pd.to_datetime(df[date_col], dayfirst=True, errors='coerce')
            else:
                df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
        else:
            df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=['Date'])
        df = df.set_index('Date')
        df = df[~df.index.duplicated(keep='last')]
        
    # Convert TMC to MCM if units column exists
    for col in ['storage', 'inflow', 'outflow']:
        matching_cols = [c for c in df.columns if col in c.lower()]
        for c in matching_cols:
            if 'tmc' in c.lower():
                df[c] = df[c] * 28.3168466  # 1 TMC = 28.3168466 MCM
                df = df.rename(columns={c: c.lower().replace('tmc', 'mcm')})
                
    # Normalize col names
    rename_map = {}
    for c in df.columns:
        if 'storage' in c.lower(): rename_map[c] = 'storage'
        if 'inflow' in c.lower(): rename_map[c] = 'inflow'
        if 'outflow' in c.lower(): rename_map[c] = 'outflow'
    df = df.rename(columns=rename_map)
    
    # Handle missing monsoon season data (drop 0s if they are likely missing)
    # Only replace zero storage — inflow/outflow can legitimately be zero in dry season
    if 'storage' in df.columns:
        df['storage'] = df['storage'].replace(0.0, np.nan)
        
        # If API does not provide Inflow or Outflow, calculate Net Inflow (Delta Storage)
        if 'inflow' not in df.columns:
            # Net inflow = change in storage
            df['inflow'] = df['storage'].diff()
            df['inflow'] = df['inflow'].fillna(0.0)
            
    df['reservoir_id'] = reservoir_id
    return df

def load_imd_gridded(nc_dir: str, basin_mask: Dict[str, list]) -> pd.DataFrame:
    """
    Load IMD gridded rainfall NetCDF files and extract basin-averaged values.
    
    Args:
        nc_dir: Directory containing NetCDF files.
        basin_mask: Mapping from basin ID to a list of (lat, lon) tuples.
        
    Returns:
        DataFrame with basin IDs as columns and Date as index.
    """
    if xr is None:
        raise ImportError("xarray is required for reading NetCDF files.")
        
    files = glob.glob(os.path.join(nc_dir, "*.nc"))
    if not files:
        return pd.DataFrame()
        
    ds = xr.open_mfdataset(files, combine='by_coords')
    
    basin_data = {}
    for basin_id, coords in basin_mask.items():
        # Extrapolate nearest points for the basin mask
        lats = xr.DataArray([c[0] for c in coords], dims="points")
        lons = xr.DataArray([c[1] for c in coords], dims="points")
        
        # Select nearest points
        subset = ds.sel(lat=lats, lon=lons, method="nearest")
        
        # Average over the points
        var_name = 'rain' if 'rain' in subset else ('rainfall' if 'rainfall' in subset else None)
        if var_name:
            # Replace -999 (No Data) with NaN to prevent skewed averages
            da = subset[var_name].where(subset[var_name] >= 0)
            series = da.mean(dim='points').to_series()
            basin_data[basin_id] = series
            
    df = pd.DataFrame(basin_data)
    df.index.name = 'Date'
    return df

def load_era5_soil_moisture(nc_file: str, lat: float, lon: float) -> pd.Series:
    """
    Load ERA5-Land soil moisture NetCDF for a specific location.
    
    Args:
        nc_file: Path to NetCDF file.
        lat: Latitude.
        lon: Longitude.
        
    Returns:
        Series indexed by date.
    """
    if xr is None:
        raise ImportError("xarray is required for reading NetCDF files.")
        
    if not os.path.exists(nc_file):
        return pd.Series()
        
    ds = xr.open_dataset(nc_file)
    subset = ds.sel(latitude=lat, longitude=lon, method="nearest")
    
    var_name = 'swvl1' if 'swvl1' in subset else list(subset.data_vars)[0]
    return subset[var_name].to_series()

def load_climate_indices(filepath: str) -> pd.DataFrame:
    """
    Load the combined ENSO and climate indices CSV file.
    
    Args:
        filepath: Path to the combined_climate_indices.csv file.
        
    Returns:
        DataFrame indexed by date, forward-filled to daily frequency.
    """
    if not os.path.exists(filepath):
        return pd.DataFrame()
        
    df = pd.read_csv(filepath)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    
    # Drop Year and Month columns as they are redundant with the Date index
    df = df.drop(columns=['Year', 'Month'], errors='ignore')
    
    # Normalize column names to uppercase for consistency with dataset.py expectations
    # (CSV may have lowercase 'oni', 'soi', 'nino34' but code expects 'ONI', 'SOI', 'NINO34')
    df.columns = df.columns.str.upper()
    
    # Climate indices are monthly. Resample and forward-fill to daily frequency
    # so they can be merged with the daily reservoir and rainfall data.
    daily_df = df.resample('D').ffill()
    
    return daily_df
