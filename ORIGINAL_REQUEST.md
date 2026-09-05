# Original User Request

## 2026-09-03T17:57:15Z

# Teamwork Project Prompt — Draft

> Status: Launched
> Goal: Craft prompt → get user approval → delegate to teamwork_preview
> Requested team: [none — teamwork routes from the description]

Project description: Extract continuous daily inflow and storage data (2010-2024) for 7 missing Indian reservoirs (Srisailam, Nagarjuna Sagar, Mettur, Jayakwadi, Ujjani, Sardar Sarovar, and Ukai) by writing Python scripts to scrape data from 4 alternative sources: UW-SASWE/RAT, reservoirs.earth, NWDP, and data.gov.in.

Working directory: c:\Users\odeda\Desktop\Projects\PBL
Integrity mode: benchmark

## Requirements

### R1. Python Extraction Pipeline
Write a Python pipeline that navigates the provided links (UW-SASWE/RAT, reservoirs.earth, nwdp.nwic.gov.in/dataset/reservoir, and data.gov.in) to locate and extract the daily historical inflow and storage data for the 7 target reservoirs.

### R2. Data Structuring
The extracted data must be cleaned and structured into CSV files containing the columns: Date, Reservoir_Name, Inflow (cusecs/cumecs), and Storage (TMC/MCM). 

### R3. File Output
The pipeline must save the final CSV files directly to the `data/raw/wris/` directory.

## Acceptance Criteria

### Execution & Output
- [ ] A main Python script executes without syntax errors.
- [ ] The script successfully outputs at least one CSV file per target reservoir into `data/raw/wris/`.
- [ ] The output CSVs contain continuous daily data within the 2010-2024 timeframe.
- [ ] The output CSVs contain valid, non-synthetic numerical data for both Inflow and Storage.

## 2026-09-03T19:08:41Z

Build a complete Spatio-Temporal Graph Neural Network (STGNN) training pipeline for 10 Peninsular Indian reservoirs using real historical daily inflow and storage data (2010-2024) already on disk at `data/raw/wris/`.

Working directory: c:\Users\odeda\Desktop\Projects\PBL
Integrity mode: benchmark

## Context

The project already has 10 raw CSVs in `data/raw/wris/` with the schema:
`Date, Reservoir_Name, Inflow (cusecs/cumecs), Storage (TMC/MCM)`.

The 10 reservoirs span 5 basins with the following known directed flow topology (upstream → downstream):
- Almatti → Srisailam → Nagarjuna Sagar (Krishna Basin)
- Krishnaraja Sagara → Mettur (Cauvery Basin)
- Tungabhadra → Srisailam (Krishna tributary)
- Ujjani → Jayakwadi (Godavari/Krishna lateral)
- Sardar Sarovar (Narmada, isolated node)
- Ukai (Tapi, isolated node)

This physical river topology must be encoded into the graph structure — edges must reflect real hydraulic connectivity, not just statistical correlation.

## Requirements

### R1. Data Preprocessing & Alignment
All 10 CSVs must be aligned to the common temporal window `2010-01-01` to `2024-12-31` (5,479 daily timesteps). Missing values must be handled without synthetic generation. Features must be normalized to prevent gradient explosion during training.

### R2. Graph Construction
Build a directed adjacency matrix (10×10) encoding the hydraulic connectivity described in the Context section above. The adjacency matrix must be justified by real river basin topology, not correlation statistics.

### R3. Sliding Window Tensor Dataset
Construct a PyTorch Dataset using a sliding window approach — input: 30-day lookback of inflow + storage across all 10 nodes; target: next 7-day inflow forecast per node. Save the compiled tensors to `data/processed/`.

### R4. STGNN Model & Training
Train a Spatio-Temporal Graph Neural Network (using graph convolution for spatial dependencies and recurrent/attention layers for temporal dependencies) that forecasts 7-day inflow for all reservoir nodes simultaneously. The model should be trained on 2010-2020 and validated on 2021-2024.

### R5. Evaluation & Results
Compute Nash-Sutcliffe Efficiency (NSE), RMSE, and MAE on the held-out 2021-2024 test split per reservoir and in aggregate. Save all metrics and produce at minimum one prediction-vs-actual plot per reservoir.

## Acceptance Criteria

### Data Pipeline
- [ ] All 10 CSVs load without errors and align to the 5,479-row common window.
- [ ] No NaN values remain in the processed tensors.
- [ ] Adjacency matrix is a (10×10) directed matrix with edges consistent with the basin topology in Context.

### Model Training
- [ ] Training completes without error across the 2010-2020 window.
- [ ] Training loss curves are saved to `outputs/training_curves/`.

### Evaluation
- [ ] Aggregate NSE on the 2021-2024 test split is >= 0.60.
- [ ] Per-reservoir NSE, RMSE, and MAE are logged to `outputs/metrics.json`.
- [ ] At least one prediction-vs-actual plot per reservoir is saved to `outputs/plots/`.

## 2026-09-05T18:32:44Z

# Teamwork Project Prompt — Draft

> Status: Launched
> Goal: Craft prompt → get user approval → delegate to teamwork_preview
> Requested team: Full team

Project description: Find and integrate real, measured historical daily inflow data for four specific Indian reservoirs (Jayakwadi, Nagarjuna Sagar, Sardar Sarovar, Mettur) from official government sources, ensuring zero synthetic generation.

Working directory: C:\Users\odeda\Desktop\Projects\PBL
Integrity mode: benchmark

## Requirements

### R1. Real Data Sourcing
Find REAL MEASURED daily inflow data for 4 Indian reservoirs (Jayakwadi, Nagarjuna Sagar, Sardar Sarovar pre-2021, and Mettur pre-2021). HARD RULES: NO synthetic data, NO random fills, NO formula-derived inflow (no water-balance ΔS+outflow computation) — only gauge/dam measurements published by an agency. Interpolation of small gaps is done downstream by existing tooling — you only fetch and format. Do NOT invent station names or use estimated/modelled inflow.

### R2. Jayakwadi Target (Godavari)
Current source "Dhalegaon" CWC gauge is unusable (80% zeros, all-zero in 2023). Hunt: (a) NWDP live catalog: package_search?q=jayakwadi and q=paithan and q=nathsagar (the web UI nwdp.nwic.gov.in may respond where the API hangs); (b) Maharashtra WRD daily dam bulletins (wrd.maharashtra.gov.in / mahagenedev — Jayakwadi / Nath Sagar Dam daily inflow reports); (c) Godavari Marathwada Irrigation Development Corporation (GMIDC) daily reports; (d) any CWC site on the Godavari upstream of Paithan with non-zero records.

### R3. Nagarjuna Sagar Target (Krishna)
Current source duplicates Srisailam's gauge on 76.6% of days. Hunt: (a) NWDP live catalog: package_search?q=nagarjuna and q=rajiv sagar and q=srisailam — AP SW Dept published a Srisailam storage dataset, check whether AP SW / Telangana SW also publish NAGARJUNA SAGAR reservoir level/storage/inflow datasets; (b) Telangana IMS/telemetry packages; (c) CWC AP/Telangana manual daily discharge stations on the Krishna BETWEEN Srisailam and Nagarjuna Sagar dams (e.g. Sripada Yellampalli, landing sites near the foreshore).

### R4. Sardar Sarovar pre-2021 Target (Narmada)
Current Mandleshwar gauge covers 2021+ only. Hunt: (a) NWDP "River Discharge CWC Madhya Pradesh (1950 - 2000) Manual Daily" and (1972-2020) archives — look for Narmada main-stem stations (Hoshangabad, Mandleshwar, Omkareshwar, Punasa, Garudeshwar) with 2010-2020 daily records; (b) NCA (Narmada Control Authority) published daily flow reports.

### R5. Mettur pre-2021 Target (Cauvery)
Current Biligundulu gauge covers 2021+ only. Hunt: Find a Cauvery main-stem station upstream of Mettur with pre-2021 records.

### R6. Data Integration & Traceability
Every dataset must record: source URL, resource/station ID, date range, unit. Nothing may overwrite existing files without a backup to `data/raw/wris/backup_pre_<name>_patch/`. If a source cannot be found for a target after honest effort, report the searches performed (queries + URLs + results) rather than forcing a patch. Do NOT touch `data/raw/wris_v2/` directly (it regenerates via build_wris_v2.py). Python environment available at `C:\Users\odeda\AppData\Local\Programs\Python\Python313\python.exe` (pandas 2.2.3, pytest, requests, xarray, netCDF4 available — no torch locally).

## Acceptance Criteria

### Execution & Integration
- [ ] Raw fetch cached under `data/raw/nwdp_cache/` or `data/raw/sources/`.
- [ ] Fetch script committed under `scripts/` (pattern: `scripts/patch_ssp_target.py`).
- [ ] `data/raw/wris/<slug>.csv` inflow replaced ONLY for dates where the new real data exists.
- [ ] Patch manifest JSON generated with source/rationale/stats (pattern: `data/raw/wris/mettur_target_patch_manifest.json`).

### Verification
- [ ] `scripts/build_wris_v2.py` successfully re-runs.
- [ ] `scripts/qa_wris_data.py --dir data/raw/wris_v2 --out runs/data_qa_report_v2.json` shows the patched reservoir with: no NaN, artifact days <= 5, seasonal JJAS share plausible, and NO all-zero years.
- [ ] `pytest tests/ -q` passes (except pre-existing torch-dependent failures).
- [ ] A summary report per reservoir is generated containing: source, rows, years covered, mean/max, and what changed.

## 2026-09-05T18:43:09Z

The user has provided additional alternative sources to check if the primary ones fail. Please relay this immediately to the active explorers:
1. UW-SASWE/RAT: https://github.com/UW-SASWE/RAT
2. Reservoirs.earth: https://reservoirs.earth/india
3. NWDP reservoir dataset: https://nwdp.nwic.gov.in/dataset/reservoir
4. Data.gov.in: https://www.data.gov.in
