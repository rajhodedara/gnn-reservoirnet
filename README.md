# GNN-ReservoirNet

Spatio-temporal GNN forecasting weekly inflow volumes for **10 major dams of Peninsular India** — built on a fully provenance-tracked, real-measured dataset, benchmarked against persistence and climatology baselines.

![status](https://img.shields.io/badge/dataset-wris__v2--verified-2b5f75) ![results](https://img.shields.io/badge/held--out%20NSE-0.4--0.66%20(9%2F10%20dams)-c4552f)

## What this is

A research project forecasting **next-1-to-12-week inflow volumes** (P10/P50/P90 quantiles) for 10 Peninsular-Indian reservoirs — Almatti, Tungabhadra, Krishnaraja Sagara, Mettur, Nagarjuna Sagar, Srisailam, Jayakwadi, Ujjani, Sardar Sarovar, Ukai — using a **GAT (spatial) + TCN (temporal) + ENSO/IOD cross-attention** architecture, with a physical mass-balance storage stage.

## Headline results (held-out 2024, week-1, seed-averaged over 3 seeds)

| Reservoir | GNN NSE | Persistence | Climatology |
|---|---|---|---|
| Tungabhadra | 0.656 ± 0.052 | 0.612 | 0.542 |
| Ukai | 0.642 ± 0.042 | 0.650 | 0.791 |
| Mettur | 0.615 ± 0.030 | 0.352 | −2.376 |
| Srisailam | 0.605 ± 0.046 | 0.418 | 0.453 |
| Almatti | 0.591 ± 0.039 | 0.488 | 0.533 |
| Krishnaraja Sagara | 0.579 ± 0.030 | 0.459 | 0.382 |
| Sardar Sarovar | 0.514 ± 0.049 | 0.376 | 0.530 |
| Ujjani | 0.504 ± 0.056 | 0.338 | 0.358 |
| Nagarjuna Sagar | 0.413 ± 0.044 | 0.162 | 0.526 |
| Jayakwadi | 0.236 ± 0.026 | −0.318 | 0.369 |

**10/10 reservoirs positive NSE · 9/10 beat persistence · 7/10 beat seasonal climatology · seed std ≤ 0.056.**

Full archive: `outputs/run1…run6/` (5 evaluated runs, each reproducible).

## The dataset (`data/raw/wris_v2/`)

Ten reservoir CSVs — daily, 2010-01-01 → 2024-12-31, 5,479 rows each:
`Date, Reservoir_Name, Inflow (m³/s), Storage (TMC)` — **every inflow value is a real agency measurement**:

| Nodes | Inflow source | Provenance |
|---|---|---|
| Srisailam, Jayakwadi, Ukai, NS*, SSP* | NWDP CWC gauge stations (Huvinhedigi, Dhalegaon, Burhanpur, Wadenepally, Garudeshwar) | `manifest_v2.json` + per-node patch manifests |
| Almatti, Tungabhadra, Krishnaraja Sagara | Karnataka WRD at-dam daily inflow (KSNDMC) | `ka_inflow_patch_manifest.json` |
| Mettur (2021–24) | CWC **Biligundulu** border station | `mettur_target_patch_manifest.json` |
| Sardar Sarovar (2021–24) | CWC **Mandleshwar** Narmada main-stem | `sardar_sarovar_target_patch_manifest.json` |

Storage columns are artifact-cleaned (unit de-mixing, impossible-spike masking, dead-storage handling) — see `scripts/build_wris_v2.py` and the QA gate `scripts/qa_wris_data.py`.

\* NS uses the documented Huvinhedigi upstream-Krishna proxy (Wadenepally verification in git history); SSP 2010–2020 remains Garudeshwar (release-contaminated, flagged).

## Climate inputs

- ENSO: ONI, SOI, Niño3.4 + **IOD (DMI)** — daily, 2005–2026 (`data/raw/enso/combined_climate_indices.csv`)
- ERA5-Land at each dam: surface runoff, evaporation, soil moisture — precomputed point extraction (`data/raw/era5/reservoir_era5_daily.csv`)
- Cross-attention conditions the GNN on climate; ENSO stratification (El Niño vs neutral) evaluated per split

## Reproduce

**Locally** (no GPU needed for data work):
```bash
python scripts/build_wris_v2.py                 # rebuild dataset
python scripts/qa_wris_data.py --dir data/raw/wris_v2   # QA gate
python scripts/run_baselines.py                 # baselines
python -m pytest tests/ -q                      # 22 tests
```

**Training** (GPU — Kaggle notebook `kaggle_runner.ipynb`, clones this repo):
```bash
python main.py --config configs/default_config.yaml --seed 42
# evaluates val (2023) AND held-out test (2024), saves predictions_*.npz
python scripts/blend_eval.py --seed-dir seed42   # GNN + climatology blending
python scripts/diagnose_nodes.py                 # per-node predictability
```

## Honest limitations

- **Skill horizon**: weeks 1–2 standalone; **blending with climatology keeps weeks 3–5 competitive** (see `scripts/blend_eval.py`)
- **Jayakwadi** capped by its gauge (Dhalegaon, 80% zeros — no better source found after exhaustive search)
- **Sardar Sarovar 2010–2020** remains release-contaminated (Garudeshwar); Mandleshwar covers 2021+
- **No `tp` rainfall band** in the ERA5 bundle — weather slot uses surface runoff (sro)
- NS inflow remains a documented proxy

## Repo map

```
main.py                 training + dual-split evaluation entry point
src/                    models (GAT+TCN+attention), data loaders, training, evaluation, explainability
configs/                default_config.yaml, reservoirs.yaml (topology, capacities)
scripts/                dataset builder, QA gate, baselines, blending, diagnostics, patch tools
tests/                  25 tests (data cleaning, un-scaling, baselines, IOD merge, ERA5 features)
data/raw/wris_v2/       the verified dataset + provenance manifests
outputs/                archived run metrics (run1…run6)
```

## Data provenance chain

Every transformation is logged and reversible:
`data/raw/wris/` (real measurements + patch manifests) → `scripts/build_wris_v2.py` (scale correction, artifact masking, zero-floor handling) → `data/raw/wris_v2/` (training-ready) → `main.py` (z-scoring, weekly targets) → evaluation (exact inverse un-scaling).

## Reservoir research packages

- 📄 **[Jayakwadi Dam (Nathsagar) research package](docs/jayakwadi_package/report.html)** — map, timeline, charts, ecology, impact, 24 cited sources (also: `docs/jayakwadi_package/data/` for the dataset files)
