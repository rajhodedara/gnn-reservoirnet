"""Download ENSO/Climate indices from NOAA PSL and BOM.

Usage:
    python scripts/download_enso_data.py

Downloads:
    - ONI (Oceanic Niño Index) from NOAA PSL
    - SOI (Southern Oscillation Index) from NOAA PSL
    - Niño 3.4 SST anomalies from NOAA PSL
    - IOD (Indian Ocean Dipole / DMI) from NOAA PSL
"""

import os
import sys
from pathlib import Path

import pandas as pd
import requests


OUTPUT_DIR = Path("data/raw/enso")

# Direct text file URLs from NOAA PSL
ENSO_SOURCES = {
    "oni": {
        "url": "https://psl.noaa.gov/data/correlation/oni.data",
        "description": "Oceanic Niño Index (3-month running mean of ERSSTv5 SST anomalies in Niño 3.4)",
        "skip_header": 1,
        "skip_footer": 8,
    },
    "soi": {
        "url": "https://psl.noaa.gov/data/correlation/soi.data",
        "description": "Southern Oscillation Index",
        "skip_header": 1,
        "skip_footer": 4,
    },
    "nino34": {
        "url": "https://psl.noaa.gov/data/correlation/nina34.anom.data",
        "description": "Niño 3.4 SST Anomalies",
        "skip_header": 1,
        "skip_footer": 4,
    },
    "iod_dmi": {
        "url": "https://psl.noaa.gov/gcos_wgsp/Timeseries/DMI/",
        "description": "Indian Ocean Dipole Mode Index (DMI)",
        "skip_header": 0,
        "skip_footer": 0,
        "is_html": True,
    },
}

MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def download_text_index(name: str, config: dict) -> pd.DataFrame | None:
    """Download a NOAA PSL text-format climate index.

    Args:
        name: Index identifier (e.g., 'oni', 'soi').
        config: Dictionary with 'url', 'skip_header', 'skip_footer' keys.

    Returns:
        DataFrame with Year and monthly columns, or None on failure.
    """
    url = config["url"]
    print(f"  Downloading {name.upper()} from {url} ...")

    if config.get("is_html"):
        print(f"  [WARN] {name.upper()} requires HTML parsing - skipping auto-download.")
        print(f"    -> Visit: {url}")
        print(f"    -> Look for 'Get Data' -> CSV download option.")
        return None

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException as e:
        print(f"  [FAIL] Failed to download {name}: {e}")
        return None

    # Parse the space-separated text format
    lines = response.text.strip().split("\n")

    # Skip header and footer lines
    skip_h = config["skip_header"]
    skip_f = config["skip_footer"]
    data_lines = lines[skip_h:]
    if skip_f > 0:
        data_lines = data_lines[:-skip_f]

    rows = []
    for line in data_lines:
        parts = line.split()
        if len(parts) < 13:
            continue
        try:
            year = int(float(parts[0]))
            values = []
            for v in parts[1:13]:
                val = float(v)
                # NOAA uses -99.99 or -999 as missing value sentinel
                if val < -90:
                    val = float("nan")
                values.append(val)
            rows.append([year] + values)
        except (ValueError, IndexError):
            continue

    if not rows:
        print(f"  [FAIL] No data parsed for {name}")
        return None

    df = pd.DataFrame(rows, columns=["Year"] + MONTH_NAMES)
    print(f"  [OK] {name.upper()}: {len(df)} years ({int(df['Year'].min())}-{int(df['Year'].max())})")
    return df


def save_index(df: pd.DataFrame, name: str, output_dir: Path) -> None:
    """Save a climate index DataFrame to CSV.

    Args:
        df: DataFrame with Year and monthly columns.
        name: Index identifier for the filename.
        output_dir: Output directory path.
    """
    filepath = output_dir / f"{name}.csv"
    df.to_csv(filepath, index=False, float_format="%.2f")
    print(f"  -> Saved: {filepath}")


def create_long_format(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """Convert wide-format (Year × Months) to long-format (Year, Month, Value).

    Args:
        df: Wide DataFrame with Year and monthly columns.
        name: Name of the index for the value column.

    Returns:
        Long-format DataFrame.
    """
    long = df.melt(id_vars=["Year"], var_name="Month", value_name=name)
    month_map = {m: i + 1 for i, m in enumerate(MONTH_NAMES)}
    long["Month_Num"] = long["Month"].map(month_map)
    long = long.sort_values(["Year", "Month_Num"]).reset_index(drop=True)
    long["Date"] = pd.to_datetime(
        long["Year"].astype(str) + "-" + long["Month_Num"].astype(str) + "-01"
    )
    return long[["Date", "Year", "Month_Num", name]].rename(
        columns={"Month_Num": "Month"}
    )


def main() -> None:
    """Download all ENSO/climate indices."""
    print("=" * 60)
    print("ENSO / Climate Index Downloader")
    print("=" * 60)

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR.resolve()}\n")

    all_long = []

    for name, config in ENSO_SOURCES.items():
        print(f"\n[{name.upper()}] {config['description']}")
        df = download_text_index(name, config)

        if df is not None:
            # Save wide format
            save_index(df, name, OUTPUT_DIR)

            # Create long format
            long_df = create_long_format(df, name)
            all_long.append(long_df)

    # Create combined climate indices file
    if all_long:
        print("\n" + "=" * 60)
        print("Creating combined climate indices file...")

        combined = all_long[0]
        for other in all_long[1:]:
            index_col = [c for c in other.columns if c not in ["Date", "Year", "Month"]][0]
            combined = combined.merge(
                other[["Date", index_col]],
                on="Date",
                how="outer",
            )

        combined = combined.sort_values("Date").reset_index(drop=True)
        combined_path = OUTPUT_DIR / "combined_climate_indices.csv"
        combined.to_csv(combined_path, index=False, float_format="%.2f")
        print(f"-> Saved: {combined_path}")
        print(f"  Rows: {len(combined)}, Columns: {list(combined.columns)}")

    # Classify ENSO phases
    oni_path = OUTPUT_DIR / "oni.csv"
    if oni_path.exists():
        print("\nClassifying ENSO phases...")
        oni_df = pd.read_csv(oni_path)
        oni_long = create_long_format(oni_df, "oni")

        def classify(val):
            if pd.isna(val):
                return "Unknown"
            if val >= 0.5:
                return "El Niño"
            elif val <= -0.5:
                return "La Niña"
            else:
                return "Neutral"

        oni_long["Phase"] = oni_long["oni"].apply(classify)
        phases_path = OUTPUT_DIR / "enso_phases.csv"
        oni_long.to_csv(phases_path, index=False, float_format="%.2f")
        print(f"-> Saved: {phases_path}")

        # Print summary of El Niño years
        el_nino_years = (
            oni_long[oni_long["Phase"] == "El Niño"]
            .groupby("Year")
            .size()
            .reset_index(name="months")
        )
        strong_el_nino = el_nino_years[el_nino_years["months"] >= 5]
        print(f"\nStrong El Niño years (>=5 months): {sorted(strong_el_nino['Year'].tolist())}")

    print("\n" + "=" * 60)
    print("Download complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
