"""Download ERA5-Land Soil Moisture and Evapotranspiration data.

Usage:
    python scripts/download_era5_data.py

Requires:
    - cdsapi package (`pip install cdsapi`)
    - Copernicus CDS API key configured in `~/.cdsapirc`
    - See: https://cds.climate.copernicus.eu/how-to-api
"""

import os
import sys
from pathlib import Path

try:
    import cdsapi
except ImportError:
    print("Error: cdsapi is not installed. Run 'pip install cdsapi'")
    sys.exit(1)


OUTPUT_DIR = Path("data/raw/era5")

# Request parameters
# Downloading 5 years of daily snapshots (12:00) to keep file sizes manageable.
# For the full 20-year history, adjust the YEARS list.
YEARS = [str(y) for y in range(2005, 2027)]
MONTHS = [f"{m:02d}" for m in range(1, 13)]
DAYS = [f"{d:02d}" for d in range(1, 32)]

# Bounding box for Peninsular India (North, West, South, East)
AREA = [23, 73, 8, 85]


def main():
    print("=" * 60)
    print("Copernicus ERA5-Land Downloader (Peninsular India)")
    print("=" * 60)

    # Check for credentials
    cds_rc = Path.home() / ".cdsapirc"
    if not cds_rc.exists():
        print("[FAIL] Missing CDS API credentials!")
        print("Please create an account at https://cds.climate.copernicus.eu/")
        print("Then create a file at ~/.cdsapirc with your url and key.")
        print(f"Expected path: {cds_rc.resolve()}")
        sys.exit(1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    nc_path = OUTPUT_DIR / "era5_land_peninsular_india.nc"

    print(f"Requesting data for years: {YEARS[0]} to {YEARS[-1]} ...")
    print("Variables: Volumetric soil water (Layer 1 & 2), Surface runoff, Total evaporation")
    print("Time: 12:00 UTC daily")
    print("Note: CDS API requests are placed in a queue. This may take several minutes/hours depending on server load.")

    try:
        c = cdsapi.Client()
        for year in YEARS:
            print(f"\nRequesting {year}...")
            out_file = OUTPUT_DIR / f"era5_land_peninsular_india_{year}.nc"
            if out_file.exists():
                print(f"Skipping {year}, already downloaded.")
                continue

            c.retrieve(
                'reanalysis-era5-land',
                {
                    'variable': [
                        'volumetric_soil_water_layer_1',
                        'volumetric_soil_water_layer_2',
                        'surface_runoff',
                        'total_evaporation',
                    ],
                    'year': [year],
                    'month': MONTHS,
                    'day': DAYS,
                    'time': '12:00',
                    'area': AREA,
                    'format': 'netcdf',
                },
                str(out_file)
            )
            print(f"[OK] Downloaded {year} to {out_file}")
            
        print("\n[OK] All requested years downloaded successfully!")
        
    except Exception as e:
        print(f"\n[FAIL] ERA5 Download failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
