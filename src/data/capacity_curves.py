import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from typing import Dict, Any, List

class CapacityCurve:
    """
    Stores elevation-area-capacity relationship for a reservoir.
    """
    def __init__(self, elevation: np.ndarray, area: np.ndarray, capacity: np.ndarray):
        """
        Args:
            elevation: Array of elevation levels (m).
            area: Array of surface areas (sq km).
            capacity: Array of storage capacities (MCM).
        """
        # Sort by capacity to ensure strictly increasing for interpolation
        sort_idx = np.argsort(capacity)
        self.capacity = capacity[sort_idx]
        self.elevation = elevation[sort_idx]
        self.area = area[sort_idx]
        
        # Remove duplicates
        self.capacity, unique_idx = np.unique(self.capacity, return_index=True)
        self.elevation = self.elevation[unique_idx]
        self.area = self.area[unique_idx]
        
        if len(self.capacity) > 1:
            self._area_interp = interp1d(self.capacity, self.area, kind='linear', fill_value='extrapolate')
            self._elev_interp = interp1d(self.capacity, self.elevation, kind='linear', fill_value='extrapolate')
        else:
            self._area_interp = lambda x: np.full_like(x, self.area[0]) if isinstance(x, np.ndarray) else self.area[0]
            self._elev_interp = lambda x: np.full_like(x, self.elevation[0]) if isinstance(x, np.ndarray) else self.elevation[0]
            
    def get_surface_area(self, storage_mcm: float) -> float:
        """
        Piecewise linear interpolation to get reservoir surface area at a given storage level.
        """
        return float(self._area_interp(storage_mcm))
        
    def get_elevation(self, storage_mcm: float) -> float:
        """
        Get water level elevation at a given storage.
        """
        return float(self._elev_interp(storage_mcm))

def load_capacity_curves(reservoirs_config: List[Dict[str, Any]]) -> Dict[str, CapacityCurve]:
    """
    Load capacity curve data for all reservoirs.
    
    Args:
        reservoirs_config: List of dicts containing reservoir config.
                           Should have 'id' and 'curve_file' keys.
                           
    Returns:
        Dictionary mapping reservoir ID to CapacityCurve object.
    """
    curves = {}
    for res in reservoirs_config:
        res_id = res['id']
        filepath = res.get('curve_file')
        
        if filepath and pd.io.common.file_exists(filepath):
            try:
                df = pd.read_csv(filepath)
                # Assumes columns 'Elevation', 'Area', 'Capacity'
                if all(col in df.columns for col in ['Elevation', 'Area', 'Capacity']):
                    curves[res_id] = CapacityCurve(
                        elevation=df['Elevation'].values,
                        area=df['Area'].values,
                        capacity=df['Capacity'].values
                    )
            except Exception as e:
                print(f"Error loading curve for {res_id}: {e}")
                
        # Default fallback if no file or loading failed
        if res_id not in curves:
            curves[res_id] = CapacityCurve(
                elevation=np.array([0.0, 100.0]),
                area=np.array([0.0, 10.0]),
                capacity=np.array([0.0, 1000.0])
            )
            
    return curves
