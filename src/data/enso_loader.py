import pandas as pd
import numpy as np
import torch
import os

def load_oni(filepath: str) -> pd.DataFrame:
    """
    Load Oceanic Nino Index from NOAA CPC.
    
    Args:
        filepath: Path to ONI CSV/text file.
        
    Returns:
        DataFrame with DatetimeIndex and 'ONI' column.
    """
    if not os.path.exists(filepath):
        return pd.DataFrame(columns=['ONI'])
    df = pd.read_csv(filepath, sep=r'\s+', header=None, names=['Year', 'Month', 'Total', 'Anomaly'])
    df['Date'] = pd.to_datetime(df[['Year', 'Month']].assign(DAY=1))
    df = df.set_index('Date')[['Anomaly']].rename(columns={'Anomaly': 'ONI'})
    return df

def load_soi(filepath: str) -> pd.DataFrame:
    """
    Load Southern Oscillation Index.
    """
    if not os.path.exists(filepath):
        return pd.DataFrame(columns=['SOI'])
    df = pd.read_csv(filepath)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    return df[['SOI']]

def load_nino34(filepath: str) -> pd.DataFrame:
    """
    Load Nino 3.4 SST anomalies.
    """
    if not os.path.exists(filepath):
        return pd.DataFrame(columns=['NINO34'])
    df = pd.read_csv(filepath)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    return df[['NINO34']]

def load_iod(filepath: str) -> pd.DataFrame:
    """
    Load Indian Ocean Dipole from BOM.
    """
    if not os.path.exists(filepath):
        return pd.DataFrame(columns=['IOD'])
    df = pd.read_csv(filepath)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.set_index('Date')
    return df[['IOD']]

def classify_enso_phase(oni_series: pd.Series) -> pd.Series:
    """
    Classify each month as El Nino / La Nina / Neutral based on ONI thresholds.
    
    Args:
        oni_series: Series containing ONI values.
        
    Returns:
        Series containing strings 'El Nino', 'La Nina', or 'Neutral'.
    """
    conditions = [
        (oni_series > 0.5),
        (oni_series < -0.5)
    ]
    choices = ['El Nino', 'La Nina']
    return pd.Series(np.select(conditions, choices, default='Neutral'), index=oni_series.index, name='Phase')

def create_climate_tensor(
    oni_path: str,
    soi_path: str,
    nino34_path: str,
    iod_path: str,
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Combine all indices into a single tensor with lag features.
    
    Args:
        oni_path: Path to ONI data.
        soi_path: Path to SOI data.
        nino34_path: Path to NINO34 data.
        iod_path: Path to IOD data.
        start_date: Start date for the merged data.
        end_date: End date for the merged data.
        
    Returns:
        Merged DataFrame with forward-filled daily values.
    """
    oni = load_oni(oni_path)
    soi = load_soi(soi_path)
    nino34 = load_nino34(nino34_path)
    iod = load_iod(iod_path)
    
    df = oni.join(soi, how='outer').join(nino34, how='outer').join(iod, how='outer')
    
    # Filter to requested date range FIRST
    df = df.loc[start_date:end_date]
    
    # Resample to daily and forward fill
    daily_idx = pd.date_range(start=start_date, end=end_date, freq='D')
    df = df.reindex(daily_idx)
    df = df.ffill().bfill().fillna(0.0)
    
    # Add phase
    if 'ONI' in df.columns:
        df['Phase'] = classify_enso_phase(df['ONI'])
        
    return df
