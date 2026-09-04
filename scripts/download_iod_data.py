import os
import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path('data/raw/climate_indices')
OUTPUT_FILE = OUTPUT_DIR / 'iod.csv'

URL = 'https://psl.noaa.gov/gcos_wgsp/Timeseries/Data/dmi.had.long.data'

def fetch_iod_data():
    print("Fetching historical IOD (DMI) data from NOAA...")
    headers = {'User-Agent': 'Mozilla/5.0'}
    
    response = requests.get(URL, headers=headers, timeout=15)
    if response.status_code != 200:
        print(f"Error fetching data: {response.status_code}")
        return
        
    lines = response.text.strip().split('\n')
    
    # First line is start and end year
    years_info = lines[0].strip().split()
    start_year = int(years_info[0])
    end_year = int(years_info[1])
    
    records = []
    
    for line in lines[1:]:
        parts = line.strip().split()
        if not parts or len(parts) < 13:
            # Reached the end metadata section
            break
            
        year = int(parts[0])
        monthly_values = [float(v) for v in parts[1:13]]
        
        for month in range(1, 13):
            # Missing values are usually -99.9 or similar, let's filter those
            val = monthly_values[month - 1]
            if val < -90:
                continue
                
            # Create a date for the first of the month
            date_str = f"{year}-{month:02d}-01"
            records.append({'Date': date_str, 'iod': val})
            
    df = pd.DataFrame(records)
    df = df.sort_values('Date')
    
    # We want data from 2005 to 2026 to match WRIS and ERA5
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[(df['Date'].dt.year >= 2005) & (df['Date'].dt.year <= 2026)]
    
    # Convert back to string for consistency
    df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"[SUCCESS] Saved {len(df)} monthly IOD records (2005-2024) to {OUTPUT_FILE}")

if __name__ == "__main__":
    fetch_iod_data()
