"""
Source Navigator Module for Indian Reservoir Pipeline.

Authoritatively navigates and probes the 4 alternative data sources:
1. UW-SASWE/RAT (https://github.com/UW-SASWE/RAT)
2. reservoirs.earth (https://reservoirs.earth)
3. nwdp.nwic.gov.in/dataset/reservoir (https://nwdp.nwic.gov.in/dataset/reservoir)
4. data.gov.in (https://data.gov.in)

Analyzes availability, response status, schema metadata, and integration feasibility.
"""

import logging
import time
from typing import Dict, Any, Optional
import requests
import urllib3

urllib3.disable_warnings()

logger = logging.getLogger(__name__)

REQUIRED_SOURCES = {
    "UW_SASWE_RAT": "https://github.com/UW-SASWE/RAT",
    "RESERVOIRS_EARTH": "https://reservoirs.earth",
    "NWDP_DATASET": "https://nwdp.nwic.gov.in/dataset/reservoir",
    "DATA_GOV_IN": "https://data.gov.in",
}

SOURCE_METADATA = {
    "UW_SASWE_RAT": {
        "name": "UW-SASWE/RAT (Reservoir Assessment Tool)",
        "url": REQUIRED_SOURCES["UW_SASWE_RAT"],
        "category": "Hydrological Modeling Code",
        "target_dams_available": False,
        "data_type": "VIC modeled runoff & satellite altimetry (2-5 day intervals)",
        "historical_range": "2015-present (multi-sensor TMS-OS)",
        "verdict": "Code repository only; zero precomputed inflow/storage series for peninsular India target dams.",
    },
    "RESERVOIRS_EARTH": {
        "name": "reservoirs.earth",
        "url": REQUIRED_SOURCES["RESERVOIRS_EARTH"],
        "category": "Open Reservoir Transparency Platform",
        "target_dams_available": True,
        "data_type": "Weekly CWC storage bulletins (Thursday snapshots)",
        "historical_range": "Recent weekly snapshots; no 2010-2024 daily continuous archive",
        "inflow_available": False,
        "verdict": "Weekly storage only; inflow completely absent; lacks 15-year continuous daily archive.",
    },
    "NWDP_DATASET": {
        "name": "National Water Data Portal - Reservoir Layer",
        "url": REQUIRED_SOURCES["NWDP_DATASET"],
        "category": "Geospatial Vector Boundary Layer",
        "target_dams_available": True,
        "data_type": "GIS vector boundaries (KML, GeoJSON, Shapefile)",
        "verdict": "Hosts dam boundary polygons; actual numerical time series reside in NWDP CKAN Datastore.",
    },
    "DATA_GOV_IN": {
        "name": "Open Government Data (OGD) India",
        "url": REQUIRED_SOURCES["DATA_GOV_IN"],
        "category": "National Data Federation Portal",
        "target_dams_available": True,
        "data_type": "Federated water catalog; requires registered API key for programmatic queries",
        "verdict": "Federates metadata to NWDP/CWC; requires API token; directs to NWDP CKAN for raw CSVs.",
    },
}


class SourceNavigator:
    """Navigates, probes, and audits connectivity to the 4 required data sources."""

    def __init__(self, user_agent: Optional[str] = None):
        self.user_agent = user_agent or (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36 ReservoirPipeline/1.0"
        )
        self.headers = {"User-Agent": self.user_agent}

    def probe_source(self, source_key: str, timeout: int = 15) -> Dict[str, Any]:
        """Probe an individual source URL and record response metrics."""
        if source_key not in REQUIRED_SOURCES:
            raise KeyError(f"Unknown source key: {source_key}. Expected one of {list(REQUIRED_SOURCES.keys())}")

        url = REQUIRED_SOURCES[source_key]
        meta = SOURCE_METADATA.get(source_key, {})
        start_time = time.time()

        try:
            resp = requests.get(url, headers=self.headers, timeout=timeout, verify=False)
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            result = {
                "source_key": source_key,
                "name": meta.get("name", source_key),
                "url": url,
                "reachable": resp.status_code in [200, 301, 302, 403],
                "status_code": resp.status_code,
                "response_time_ms": elapsed_ms,
                "content_type": resp.headers.get("Content-Type", ""),
                "content_length": len(resp.content),
                "metadata": meta,
                "error": None,
            }
            logger.info(f"Probed {source_key} ({url}): HTTP {resp.status_code} in {elapsed_ms}ms")
            return result
        except requests.exceptions.Timeout as e:
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            logger.warning(f"Timeout probing {source_key} ({url}) after {elapsed_ms}ms: {e}")
            return {
                "source_key": source_key,
                "name": meta.get("name", source_key),
                "url": url,
                "reachable": False,
                "status_code": None,
                "response_time_ms": elapsed_ms,
                "content_type": "",
                "content_length": 0,
                "metadata": meta,
                "error": f"Timeout: {e}",
            }
        except requests.exceptions.RequestException as e:
            elapsed_ms = round((time.time() - start_time) * 1000, 2)
            logger.warning(f"Error probing {source_key} ({url}): {e}")
            return {
                "source_key": source_key,
                "name": meta.get("name", source_key),
                "url": url,
                "reachable": False,
                "status_code": None,
                "response_time_ms": elapsed_ms,
                "content_type": "",
                "content_length": 0,
                "metadata": meta,
                "error": str(e),
            }

    def probe_all_sources(self, timeout: int = 15) -> Dict[str, Dict[str, Any]]:
        """Probes all 4 required sources and returns comprehensive status report."""
        results = {}
        for source_key in REQUIRED_SOURCES:
            results[source_key] = self.probe_source(source_key, timeout=timeout)
        return results

    def get_source_metadata(self, source_key: Optional[str] = None) -> Any:
        """Retrieve curated metadata and architectural assessment for data sources."""
        if source_key is not None:
            return SOURCE_METADATA.get(source_key, {})
        return SOURCE_METADATA

    def generate_probe_report(self, probe_results: Optional[Dict[str, Dict[str, Any]]] = None) -> str:
        """Generates formatted string summary of probed sources."""
        results = probe_results or self.probe_all_sources()
        lines = [
            "=================================================================",
            " MULTI-SOURCE NAVIGATION & AUDIT REPORT",
            "=================================================================",
        ]
        for key, res in results.items():
            status = f"HTTP {res['status_code']}" if res['status_code'] else f"FAILED ({res['error']})"
            reach = "[ONLINE]" if res['reachable'] else "[OFFLINE]"
            lines.append(f"{reach} {res['name']}")
            lines.append(f"  URL: {res['url']}")
            lines.append(f"  Status: {status} | Latency: {res['response_time_ms']}ms")
            lines.append(f"  Role: {res['metadata'].get('category', 'N/A')}")
            lines.append(f"  Assessment: {res['metadata'].get('verdict', 'N/A')}")
            lines.append("-" * 65)
        return "\n".join(lines)
