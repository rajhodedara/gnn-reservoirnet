"""
NWDP CKAN Extractor Module for Indian Reservoir Pipeline.

Extracts genuine historical daily reservoir storage and river discharge (inflow)
observations from the National Water Data Portal (NWDP / NWIC) CKAN Datastore API.

Authoritative Resources on NWDP:
1. Srisailam Storage: AP SW Department (Resource: be847b75-154e-4cc8-b4ff-f56ad8735644)
2. CWC Gujarat River Discharge (Sardar Sarovar - Garudeshwar, Ukai - GHALA):
   Resource: b076861e-1513-410e-bafa-a1e34cd8f493
3. CWC Madhya Pradesh River Discharge (Ukai - BURHANPUR):
   Resource: 5708264d-5aea-4e39-8e64-e837f55d4c1b
4. CWC Maharashtra River Discharge (Jayakwadi - Dhalegaon, Ujjani - Dhond):
   Resource: 9c659865-ab21-4ffa-a3f9-edbae14f5c86
5. CWC Karnataka River Discharge (Krishna/Srisailam - Huvinhedigi, Bhima/Ujjani - Yadgir):
   Resource: f95150ea-c8fc-4740-8815-d9c34c9d53a3
6. CWC Tamil Nadu River Discharge (Mettur - KODUMUDI / MUSIRI):
   Resource: fca9df0b-47b1-4f1a-8e59-1b43a8c0ae73
7. CWC Telangana River Discharge (Nagarjuna Sagar - VEERLAPALEM):
   Resource: 1b9088b5-d196-4c5d-8780-a888e7e9e86b
"""

import os
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests
import pandas as pd
import urllib3

urllib3.disable_warnings()

logger = logging.getLogger(__name__)

NWDP_BASE_URL = "https://nwdp.nwic.gov.in/api/3/action/"
DATASTORE_SEARCH_URL = f"{NWDP_BASE_URL}datastore_search"

# Curated CWC / State Station Mappings
STATION_RESOURCES = {
    "srisailam_storage": {
        "resource_id": "be847b75-154e-4cc8-b4ff-f56ad8735644",
        "station": "SRI SAILAM PROJECT",
        "value_col": "Manual Daily Reservoir storage (mcm)",
        "unit": "MCM",
        "type": "storage",
    },
    "srisailam_inflow": {
        "resource_id": "f95150ea-c8fc-4740-8815-d9c34c9d53a3",
        "station": "Huvinhedigi",
        "river": "Krishna",
        "value_col": "Manual Daily River Water Discharge (m3/sec)",
        "unit": "cumecs",
        "type": "inflow",
    },
    "nagarjuna_sagar_inflow": {
        "resource_id": "1b9088b5-d196-4c5d-8780-a888e7e9e86b",
        "station": "VEERLAPALEM",
        "river": "Krishna",
        "value_col": "Manual Daily River Water Discharge (m3/sec)",
        "unit": "cumecs",
        "type": "inflow",
        "backup_resource_id": "f95150ea-c8fc-4740-8815-d9c34c9d53a3",
        "backup_station": "Huvinhedigi",
    },
    "mettur_inflow": {
        "resource_id": "fca9df0b-47b1-4f1a-8e59-1b43a8c0ae73",
        "station": "KODUMUDI",
        "river": "Cauvery",
        "value_col": "Manual Daily River Water Discharge (m3/sec)",
        "unit": "cumecs",
        "type": "inflow",
    },
    "jayakwadi_inflow": {
        "resource_id": "9c659865-ab21-4ffa-a3f9-edbae14f5c86",
        "station": "Dhalegaon",
        "river": "Godavari",
        "value_col": "Manual Daily River Water Discharge (m3/sec)",
        "unit": "cumecs",
        "type": "inflow",
    },
    "ujjani_inflow": {
        "resource_id": "f95150ea-c8fc-4740-8815-d9c34c9d53a3",
        "station": "Yadgir",
        "river": "Bhima",
        "value_col": "Manual Daily River Water Discharge (m3/sec)",
        "unit": "cumecs",
        "type": "inflow",
        "secondary_station": "Dhond",
        "secondary_resource_id": "9c659865-ab21-4ffa-a3f9-edbae14f5c86",
    },
    "sardar_sarovar_inflow": {
        "resource_id": "b076861e-1513-410e-bafa-a1e34cd8f493",
        "station": "Garudeshwar",
        "river": "Narmada",
        "value_col": "Manual Daily River Water Discharge (m3/sec)",
        "unit": "cumecs",
        "type": "inflow",
    },
    "ukai_inflow": {
        "resource_id": "5708264d-5aea-4e39-8e64-e837f55d4c1b",
        "station": "BURHANPUR",
        "river": "Tapi",
        "value_col": "Manual Daily River Water Discharge (m3/sec)",
        "unit": "cumecs",
        "type": "inflow",
    },
}


class NWDPExtractor:
    """Extracts reservoir storage and river discharge records from NWDP CKAN API."""

    def __init__(self, cache_dir: Optional[str] = None):
        self.cache_dir = Path(cache_dir or "data/raw/nwdp_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) ReservoirPipeline/1.0",
            "Accept": "application/json",
        })

    def query_datastore(
        self,
        resource_id: str,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 10000,
        offset: int = 0,
        timeout: int = 25,
    ) -> Dict[str, Any]:
        """Perform datastore_search query with filters."""
        params: Dict[str, Any] = {
            "resource_id": resource_id,
            "limit": limit,
            "offset": offset,
        }
        if filters:
            params["filters"] = json.dumps(filters)

        resp = self.session.get(DATASTORE_SEARCH_URL, params=params, verify=False, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        if not data.get("success"):
            raise ValueError(f"CKAN datastore_search query failed: {data}")
        return data

    def fetch_station_records(
        self,
        resource_id: str,
        station_name: str,
        max_records: int = 20000,
    ) -> List[Dict[str, Any]]:
        """Fetch all records for a given station, caching to disk for speed and reliability."""
        safe_stn = station_name.replace(" ", "_").replace("/", "_").lower()
        cache_file = self.cache_dir / f"{resource_id[:8]}_{safe_stn}.json"

        # Check local cache first
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    records = json.load(f)
                logger.info(f"Loaded {len(records)} records from cache: {cache_file.name}")
                return records
            except Exception as e:
                logger.warning(f"Error reading cache {cache_file}: {e}, falling back to live API")

        # Query live API
        records: List[Dict[str, Any]] = []
        offset = 0
        limit = 10000

        while len(records) < max_records:
            try:
                data = self.query_datastore(
                    resource_id=resource_id,
                    filters={"Station": station_name},
                    limit=limit,
                    offset=offset,
                )
                batch = data.get("result", {}).get("records", [])
                if not batch:
                    break
                records.extend(batch)
                total = data.get("result", {}).get("total", len(records))
                if len(records) >= total or len(batch) < limit:
                    break
                offset += len(batch)
            except Exception as e:
                logger.warning(f"Failed querying {station_name} from resource {resource_id}: {e}")
                break

        # Save to local cache
        if records:
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(records, f)
                logger.info(f"Cached {len(records)} records to {cache_file.name}")
            except Exception as e:
                logger.warning(f"Failed to write cache {cache_file}: {e}")

        return records

    def fetch_srisailam_storage_df(self) -> pd.DataFrame:
        """Fetch daily storage observations for Srisailam from AP SW manual package."""
        cfg = STATION_RESOURCES["srisailam_storage"]
        records = self.fetch_station_records(cfg["resource_id"], cfg["station"])
        if not records:
            logger.warning("No Srisailam storage records fetched from NWDP!")
            return pd.DataFrame(columns=["Date", "storage_mcm"])

        df = pd.DataFrame(records)
        df["Date"] = pd.to_datetime(df["Data Acquisition Time"], format="%d-%m-%Y %H:%M", errors="coerce").dt.strftime("%Y-%m-%d")
        df["storage_mcm"] = pd.to_numeric(df[cfg["value_col"]], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date")
        return df[["Date", "storage_mcm"]]

    def fetch_inflow_df(self, station_key: str) -> pd.DataFrame:
        """Fetch daily river discharge (inflow) observations for given station key."""
        if station_key not in STATION_RESOURCES:
            raise KeyError(f"Unknown station key: {station_key}")

        cfg = STATION_RESOURCES[station_key]
        records = self.fetch_station_records(cfg["resource_id"], cfg["station"])
        
        # If primary has limited rows and backup exists, fetch backup
        if cfg.get("backup_resource_id") and cfg.get("backup_station") and len(records) < 5000:
            backup_records = self.fetch_station_records(cfg["backup_resource_id"], cfg["backup_station"])
            if backup_records:
                # Combine primary with backup for missing dates
                df_prim = pd.DataFrame(records) if records else pd.DataFrame()
                df_back = pd.DataFrame(backup_records)
                
                df_back["Date"] = pd.to_datetime(df_back["Data Acquisition Time"], format="%d-%m-%Y %H:%M", errors="coerce").dt.strftime("%Y-%m-%d")
                df_back["inflow_cumecs"] = pd.to_numeric(df_back[cfg["value_col"]], errors="coerce")
                
                if not df_prim.empty:
                    df_prim["Date"] = pd.to_datetime(df_prim["Data Acquisition Time"], format="%d-%m-%Y %H:%M", errors="coerce").dt.strftime("%Y-%m-%d")
                    df_prim["inflow_cumecs"] = pd.to_numeric(df_prim[cfg["value_col"]], errors="coerce")
                    combined = pd.concat([df_prim, df_back]).dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date", keep="first")
                    return combined[["Date", "inflow_cumecs"]]
                else:
                    return df_back.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date")[["Date", "inflow_cumecs"]]

        if not records:
            logger.warning(f"No records fetched for {station_key}")
            return pd.DataFrame(columns=["Date", "inflow_cumecs"])

        df = pd.DataFrame(records)
        df["Date"] = pd.to_datetime(df["Data Acquisition Time"], format="%d-%m-%Y %H:%M", errors="coerce").dt.strftime("%Y-%m-%d")
        df["inflow_cumecs"] = pd.to_numeric(df[cfg["value_col"]], errors="coerce")
        df = df.dropna(subset=["Date"]).sort_values("Date").drop_duplicates("Date")
        return df[["Date", "inflow_cumecs"]]
