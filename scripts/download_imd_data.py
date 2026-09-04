"""Download IMD Gridded Rainfall and Temperature data.

Usage:
    python scripts/download_imd_data.py

Downloads:
    - 0.25 deg Gridded Rainfall (2005-2023)
    - Extracts data for Peninsular India (lat 8-23, lon 73-85)
    - Saves to NetCDF and CSV
"""

import os
import sys
from pathlib import Path

try:
    import imdlib as imd
except ImportError:
    print("Error: imdlib is not installed. Run 'pip install imdlib'")
    sys.exit(1)

try:
    import xarray as xr
except ImportError:
    print("Error: xarray is not installed. Run 'pip install xarray netCDF4'")
    sys.exit(1)


OUTPUT_DIR = Path("data/raw/imd")
START_YR = 2005
# Note: IMD data for the current/previous year might not be fully available.
# 2023 is usually safe.
END_YR = 2023

# Bounding box for Peninsular India
LAT_MIN, LAT_MAX = 8.0, 23.0
LON_MIN, LON_MAX = 73.0, 85.0

def main():
    print("=" * 60)
    print("IMD Gridded Data Downloader (Peninsular India)")
    print("=" * 60)

    # Create output directories
    rain_dir = OUTPUT_DIR / "rainfall"
    rain_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading rainfall data ({START_YR}-{END_YR}) to {rain_dir.resolve()} ...")
    print("Note: This will download daily 0.25 degree binary files from IMD Pune.")
    
    try:
        print("Downloading files from IMD...")
        imd.get_data('rain', START_YR, END_YR, fn_format='yearwise', file_dir=str(rain_dir))
        print("Download complete. Opening data...")
        # Load the downloaded data
        data = imd.open_data('rain', START_YR, END_YR, 'yearwise', str(rain_dir))
        
        if data is None:
            print("[FAIL] Could not load IMD data.")
            return

        print("[OK] Data downloaded and loaded successfully.")
        
        # Convert to xarray Dataset
        print("Converting to xarray dataset...")
        ds = data.get_xarray()
        
        # Extract for Peninsular India
        print(f"Subsetting for Peninsular India (Lat: {LAT_MIN}-{LAT_MAX}, Lon: {LON_MIN}-{LON_MAX})...")
        # Note: IMD coordinates might need exact slicing or nearest depending on resolution.
        # 0.25 deg resolution.
        ds_peninsular = ds.sel(lat=slice(LAT_MIN, LAT_MAX), lon=slice(LON_MIN, LON_MAX))
        
        # Save to NetCDF
        nc_path = OUTPUT_DIR / "imd_rainfall_peninsular.nc"
        print(f"Saving to NetCDF: {nc_path} ...")
        ds_peninsular.to_netcdf(nc_path)
        print(f"[OK] Saved {nc_path}")
        
        # Optionally, we can also extract timeseries for specific reservoir coordinates if needed later,
        # but storing the NetCDF is best for spatial features (GNN nodes).
        
    except Exception as e:
        print(f"[FAIL] Error processing IMD data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
