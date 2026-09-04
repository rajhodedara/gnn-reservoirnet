import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

OUTPUT_DIR = Path('data/raw/wris')

RESERVOIRS = [
    "Nagarjuna Sagar", "Srisailam", "Almatti", "Tungabhadra", "Ujjani",
    "Mettur", "Krishnaraja Sagara", "Jayakwadi", "Sardar Sarovar", "Ukai"
]

def format_filename(name):
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "") + ".csv"

def generate_synthetic_data(start_year=2005, end_year=2024):
    print("============================================================")
    print("Generating Synthetic WRIS Reservoir Data")
    print("============================================================")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Time variable for sine waves (1 year period)
    t = np.arange(len(date_range))
    annual_cycle = np.sin(2 * np.pi * t / 365.25 - np.pi/2) # Peak in summer/monsoon
    
    for res in RESERVOIRS:
        # Base realistic volumes in MCM
        base_storage = np.random.uniform(500, 2000)
        base_flow = np.random.uniform(50, 200)
        
        # Add noise and seasonal patterns
        inflow = np.maximum(0, base_flow + (base_flow * 2) * annual_cycle + np.random.normal(0, base_flow/2, len(t)))
        outflow = np.maximum(0, base_flow + (base_flow * 1.5) * annual_cycle + np.random.normal(0, base_flow/3, len(t)))
        
        # Cumulative storage integration (bounded)
        storage = np.zeros(len(t))
        storage[0] = base_storage
        for i in range(1, len(t)):
            # storage change = inflow - outflow
            storage[i] = storage[i-1] + (inflow[i] - outflow[i])
            # Bound storage between 10% and 200% of base
            storage[i] = np.clip(storage[i], base_storage * 0.1, base_storage * 2.0)
            
        df = pd.DataFrame({
            'Date': date_range.strftime('%Y-%m-%d'),
            'inflow': np.round(inflow, 2),
            'outflow': np.round(outflow, 2),
            'storage': np.round(storage, 2)
        })
        
        out_file = OUTPUT_DIR / format_filename(res)
        df.to_csv(out_file, index=False)
        print(f"[OK] Generated {len(df)} days of synthetic data for {res}")

if __name__ == "__main__":
    generate_synthetic_data()
    print("\nSynthetic data generation complete. Saved to data/synthetic/")
