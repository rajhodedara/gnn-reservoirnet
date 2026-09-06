import os, json, time, requests, re, shutil
import pandas as pd
import numpy as np

PROJECTS = {
    1: 'Kotipalli Vagu', 4: 'Priyadarshini Jurala Project', 6: 'Koil Sagar Project',
    7: 'OkaChettiVagu', 8: 'Tungabadra HLC', 10: 'Bhairavanitippa Reservoir Project',
    11: 'Rajoli Banda Diversion Scheme', 12: 'Sunkesula Project', 
    13: 'Sanjeeviah Sagar Gajuladinne Project', 16: 'Telugu Ganga Project',
    19: 'Neelam Sanjeeva Reddy Srisailam Project', 21: 'Dindi Project',
    22: 'Alimineti Madhava Reddy Project', 24: 'Nagarjuna Sagar Project',
    25: 'Musi Project', 28: 'KL Rao Sagar- Pulichinthala Project', 29: 'Palair',
    31: 'Pakhal lake', 32: 'Muniyeru Project', 33: 'Wyra Project', 34: 'Lanka Sagar',
    37: 'Prakasam Barrage', 38: 'Tungabadra LLC', 41: 'Chennai Water Suppy',
    44: 'NSLC 21st MBC @ km 101.362 (AP/TS Boarder)'
}

# Almatti, Narayanpur, Sripada Yellampalli were not found in KRMB dropdown list.

TARGETS = {'srisailam': 19, 'nagarjuna_sagar': 24}

def parse_krmb_month(project_id, month, year, session):
    url = "http://krmb.gov.in/krmb/getddAvgReport"
    params = {"projectId": str(project_id), "month": str(month), "year": str(year)}
    try:
        r = session.get(url, params=params, timeout=15)
        r.raise_for_status()
    except Exception as e:
        return []
    tables = re.findall(r'<table.*?>.*?</table>', r.text, re.DOTALL | re.IGNORECASE)
    if len(tables) < 2: return []
    rows = re.findall(r'<tr.*?>.*?</tr>', tables[1], re.DOTALL | re.IGNORECASE)
    records = []
    for row in rows:
        cells = re.findall(r'<t[dh].*?>(.*?)</t[dh]>', row, re.DOTALL | re.IGNORECASE)
        clean = [re.sub(r'<.*?>|\s+', ' ', c).strip().replace(',', '') for c in cells]
        if clean and clean[0].isdigit():
            try:
                # Srisailam and Nagarjuna Sagar format is assumed to be similar to Srisailam.
                records.append({
                    "date": clean[1],
                    "level_ft": float(clean[2]) if clean[2] else None,
                    "storage_tmc": float(clean[3]) if clean[3] else None,
                    "inflow_cusecs": float(clean[4]) if clean[4] else None,
                    "evaporation_cusecs": float(clean[-2]) if clean[-2] else None,
                    "total_outflow_cusecs": float(clean[-1]) if clean[-1] else None,
                    "raw_cells": clean
                })
            except Exception:
                pass
    return records

def fetch_data():
    session = requests.Session()
    session.headers.update({'User-Agent': 'Mozilla/5.0'})
    for slug, pid in TARGETS.items():
        print(f"Fetching {slug}...")
        for year in range(2010, 2025):
            for month in range(1, 13):
                cache_path = f"data/raw/sources/krmb/{slug}/{year}-{month:02d}.json"
                if os.path.exists(cache_path): continue
                recs = parse_krmb_month(pid, month, year, session)
                if recs:
                    with open(cache_path, 'w') as f:
                        json.dump(recs, f)
                    print(f"  Saved {slug} {year}-{month:02d} ({len(recs)} records)")
                time.sleep(0.5)

if __name__ == "__main__":
    fetch_data()
