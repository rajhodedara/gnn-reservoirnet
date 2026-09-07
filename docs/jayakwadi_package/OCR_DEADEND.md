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
