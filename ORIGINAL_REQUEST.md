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

