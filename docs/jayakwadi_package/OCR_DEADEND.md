# Jayakwadi Reservoir Inflow: PDF OCR Dead-End Report

## Overview
This report documents the exhaustive search for static, archival PDFs (annual water year books, weekly reservoir bulletins, daily gauge reports) containing daily or monthly measured inflow/level data for the Jayakwadi (Paithan) reservoir spanning the 2010-2024 water years.

The objective was to download these PDFs, OCR the inflow tables, and build a patch CSV. However, locating a viable 3-year static PDF archive proved unsuccessful.

## Sources Investigated and Failure Reasons

### 1. Maharashtra Water Resources Department (wrd.maharashtra.gov.in)
*   **Target:** Daily Reservoir Storage Reports / State Flood Status Reports (`आजचा पाणीसाठा`, `राज्य पूरस्थिती अहवाल`).
*   **Result & Failure:** The portal actively publishes these reports, and our scraper successfully parsed the `ViewPDFList` endpoints. However, the system only retains the reports for the last **~20 days** (e.g., late August to September 2026 at the time of scraping). There is no exposed static archive or directory index spanning back to 2010-2024. Additionally, legacy reports encountered character map encoding issues (`cp1252` vs Marathi unicode) which could be bypassed, but the core issue is the **missing historical volume**.

### 2. Central Water Commission (cwc.gov.in)
*   **Target:** Weekly Reservoir Level & Storage Bulletins / Water Year Books (Godavari Basin).
*   **Result & Failure:** 
    *   *Weekly Bulletins:* We successfully scraped the CWC portal and downloaded recent bulletins (e.g., `fb-27032025-new.pdf`). Analysis of the PDF via `pdfplumber` confirmed that these reports focus on aggregated live storage metrics (FRL, MWL, % of normal storage) for basin comparisons, **not direct daily inflow measurements** for specific reservoirs like Jayakwadi.
    *   *Water Year Books:* Searches across CWC publications and external search engines (DuckDuckGo) confirmed the existence of these books (Volume 1 for Daily Discharge). However, no digitized, openly accessible PDFs for the 2010-2024 period could be located. 

### 3. Godavari Marathwada Irrigation Development Corporation (gmidc.maharashtra.gov.in)
*   **Target:** Regional division responsible for Jayakwadi operations.
*   **Result & Failure:** **DNS Resolution Failure.** The domain `gmidc.maharashtra.gov.in` is currently unreachable (`Errno 11001`), making any archival PDFs hosted on their specific subdomain inaccessible.

## Untried Source & Recommendation

**The Single Source Untried:** 
The **India-WRIS (Water Resources Information System)** Data Portal API.

**Why it remains untried and why it is the best path forward:** 
The instructions explicitly constrained the search to locating and downloading "archival PDFs" for OCR. India-WRIS does possess extensive historical reservoir level and inflow data, but it is served via a dynamic web application powered by backend REST/GraphQL APIs, rather than static downloadable PDFs. Since the PDF OCR avenue is blocked by data-retention policies and inaccessible domains, intercepting the network requests on the India-WRIS portal or utilizing their telemetry API is the most viable method to extract the 2010-2024 daily inflow telemetry for Jayakwadi.

## Addendum: India-WRIS API Extraction Attempt

### API Discovery Log
We attempted to intercept the India-WRIS backend API (e.g., `https://indiawris.gov.in/wris/` and `https://nwic.gov.in/`) to extract the reservoir telemetry programmatically. 

*   **Network Probe:** All direct HTTPS/HTTP requests to `indiawris.gov.in` (IP: 164.100.85.36) resulted in hard connection timeouts (`curl: (28) Connection timed out after 21117 ms`). The NWIC/NIC infrastructure appears to actively drop packets from international cloud/datacenter IP ranges, making the portal completely unreachable from this execution environment.
*   **Alternative Endpoints:** Probed `nwic.gov.in` which was accessible, but it only contains public notices and static links to the (blocked) `indiawris.gov.in` subdomains. No functional API endpoints for reservoir data were exposed on the accessible domain.

### Gate Verdict: FAILED
Because the API servers are inaccessible, the data retrieval rate is 0% (100% missing). This fails the established gate constraint of requiring missing/zero values to be `< 60%`. Consequently, building a patch CSV via this automated pipeline is impossible. The current `jayakwadi.csv` series (~80% zeros) remains the baseline until the WRIS portal can be scraped from an unblocked residential Indian IP.


## Addendum: Maharashtra State Portals (Vector 2)

### s.mahawrd.org WAF/IP Block Assessment
*   **Target:** https://rs.mahawrd.org/ which was assumed to contain daily dynamic status reports.
*   **Result & Failure:** The ConnectionResetError and 436 status codes were not the result of a sophisticated WAF or IP ban. The domain s.mahawrd.org has expired and is currently a parked domain squatting on Sedo Domain Parking. There is no portal here.

### wrd.maharashtra.gov.in and gidcdashboard.in Scraping
*   **Target:** Flood Control Bulletins and GMIDC Daily Reservoir Data.
*   **Result & Failure:** A deep scraper was built to recursively parse ViewPDFList endpoints across the official WRD portal. We successfully extracted the Godavari/Jayakwadi PURNIYANTRAN_BOOK_GMIDC.pdf (Flood Control Booklet) and JID Paithan.pdf. 
    *   Analysis of these PDFs via PyMuPDF confirmed they are static manuals and farmer irrigation ledgers (474 pages of individual permit holders), not daily telemetry reports.
    *   Network interception of gidcdashboard.in confirmed it is a static React UI gallery with no live daily storage API routes.

### Gate Verdict: FAILED
Because the domain is parked and the available state PDFs are static procedural manuals rather than daily logs, the data retrieval rate is 0%. This fails the established gate constraint. Do not build a patch from failing data. The satellite bathymetry proxy (Earth Engine SAR) is the only remaining viable vector for purely physical data.

