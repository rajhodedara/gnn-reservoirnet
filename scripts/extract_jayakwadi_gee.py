
import ee
import pandas as pd

# [N] Neon v4.3 - GEE Sentinel-1 SAR Bathymetry Extractor
# Bypasses local WAF blocks by pulling physical measurements straight from orbit.

def run_extraction():
    print("[*] Initializing Google Earth Engine...")
    try:
        ee.Initialize()
    except Exception as e:
        print("[!] Auth token not found. Opening browser for authentication...")
        ee.Authenticate()
        ee.Initialize()

    print("[*] Earth Engine initialized. Building computational graph...")

    jayakwadi_poly = ee.Geometry.Polygon(
      [[[75.20, 19.40],
        [75.50, 19.40],
        [75.50, 19.65],
        [75.20, 19.65]]]
    )

    def get_water_area(image):
        vv = image.select("VV")
        water_mask = vv.lt(-16).rename("water")
        water_area = water_mask.multiply(ee.Image.pixelArea()).divide(1e6)
        stats = water_area.reduceRegion(
            reducer=ee.Reducer.sum(),
            geometry=jayakwadi_poly,
            scale=30,
            maxPixels=1e9
        )
        return image.set("water_area_sqkm", stats.get("water")).set("date", image.date().format("YYYY-MM-dd"))

    print("[*] Querying Sentinel-1 archive (2014 - 2024)...")
    s1_collection = ee.ImageCollection("COPERNICUS/S1_GRD") \
        .filterBounds(jayakwadi_poly) \
        .filterDate("2014-10-01", "2024-01-01") \
        .filter(ee.Filter.listContains("transmitterReceiverPolarisation", "VV")) \
        .filter(ee.Filter.eq("instrumentMode", "IW"))

    water_ts = s1_collection.map(get_water_area)

    print("[*] Executing reduction on Google servers. This will take a minute...")
    data_list = water_ts.reduceColumns(ee.Reducer.toList(2), ["date", "water_area_sqkm"]).values().get(0).getInfo()

    df = pd.DataFrame(data_list, columns=["Date", "Water_Area_sqkm"])
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.groupby("Date").mean().reset_index()
    df.sort_values("Date", inplace=True)

    output_file = "data/raw/wris_v2/Jayakwadi_Water_Area_TS_2014_2024.csv"
    df.to_csv(output_file, index=False)
    print(f"[+] Success. Extracted {len(df)} orbital observations. Saved to {output_file}")

if __name__ == "__main__":
    run_extraction()

