# HANDOFF — Jayakwadi research package

## What this package is

A self-contained research + data bundle about **Jayakwadi Dam (Nathsagar), Paithan, Maharashtra** — built 2026-09-06 as part of the GNN-ReservoirNet forecasting project. Everything is reproducible from the listed sources.

## Contents

```
jayakwadi_package/
├── report.html                                        ← the visual deliverable (open in any browser)
├── data/
│   ├── jayakwadi_facts.csv                            ← 24 key facts, each with source(s) + reliability + retrieval date
│   ├── storage_latest.csv                             ← latest publicly reported storage figures (with URLs + retrieval dates)
│   └── jayakwadi_inflow_storage_daily_2010_2024.csv   ← the compiled daily series (inflow m³/s + storage TMC) used by the forecasting model
├── SOURCES.md                                          ← bibliography (24 sources) with reliability notes
├── README.md                                           ← package index
└── HANDOFF.md                                          ← this file
```

## How to update it next season

1. **Latest storage**: open https://mwrdpravah.in/damsafety/control/pdfLatestReportEng (Maharashtra WRD dam-safety live report — updates daily ~06:49 AM). Find the `Paithan (Jayakwadi)` row. Record: date, today's live storage (Mcum), live capacity (2,170.93 Mcum), designed storage (2,909.04 Mcum). Add a row to `data/storage_latest.csv`.
   - NOTE: the PDF text layer lists two candidate values (e.g. 738.11 / 1,977.57). Confirm the column mapping on the portal once, then record the right one.
2. **Season news**: search Times of India / Hindustan Times / Indian Express Aurangabad editions for "Jayakwadi" at the end of each monsoon (Sep-Oct) and each summer (May). Add drought/overflow milestones to `storage_latest.csv` and the timeline in `report.html`.
3. **Rebuild the charts**: the four charts in `report.html` derive from the compiled daily series. If the series is extended (new years), update the SVG bars: the chart values are stored in `PBL/scratch/jayakwadi_chart_data.json` and the generator is `PBL/scratch/build_jayakwadi_report.py`.
4. **If a better inflow source is found** (e.g. Maharashtra WRD daily Jayakwadi inflow tables): follow the pattern of `scripts/patch_ssp_target.py` in the repo — backup, patch `data/raw/wris/jayakwadi.csv`, rebuild with `scripts/build_wris_v2.py`, re-run the QA gate (`scripts/qa_wris_data.py --dir data/raw/wris_v2`), commit.

## Open questions / data gaps

| Gap | Why it matters | Where to look next |
|-----|----------------|--------------------|
| Original displacement figures (1965–1976 submergence) | The social cost of the dam is under-documented online | Maharashtra WRD archives; district gazetteer of Aurangabad; NCA/academic theses on Marathwada irrigation |
| WRD live-report column mapping (738.11 vs 1,977.57 Mcum) | Needed to state "today's storage" with confidence | Open the PDF on the portal and read the table headers visually |
| Sanctuary area discrepancy (341 km² vs 225 km²) | Two official-leaning sources disagree | KBA factsheet cites the Irrigation Dept notification — request the gazette notification text |
| Species count range (234 / 251 / 300+) | Counts vary by survey year/method | The 2023 forest-clearance conservation assessment [10] is the most formal — treat as the floor |
| A non-zero inflow source for the model | Jayakwadi is the only node whose training target is dominated by zeros (the Dhalegaon gauge) | Maharashtra WRD daily bulletins (see #1); GMIDC records; T1 hunt continues |
| ERA5 `tp` (true rainfall) band | The weather slot currently uses surface runoff (sro) | Copernicus CDS API (free account) — download `total_precipitation` 2010–2024 for 73–85E / 8–23N, then re-run `scripts/extract_era5_points.py` |

## Relation to the forecasting model

The compiled daily series (`data/jayakwadi_inflow_storage_daily_2010_2024.csv`) IS the training/evaluation target for the Jayakwadi node of the GNN (`data/raw/wris_v2/jayakwadi.csv` in the repo). Its known weakness — 80% zero days (drought reality + gauge limitation) — is why the Jayakwadi node scores NSE 0.24 while most others score 0.4–0.66. Any real inflow source found above directly improves the model.
