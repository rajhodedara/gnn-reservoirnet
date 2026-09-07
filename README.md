# GNN-ReservoirNet

Spatio-temporal GNN forecasting weekly inflow volumes for **10 major dams of Peninsular India** — built on a fully provenance-tracked, real-measured dataset, benchmarked against persistence and climatology baselines.

![status](https://img.shields.io/badge/dataset-wris__v2--verified-2b5f75) ![results](https://img.shields.io/badge/held--out%20NSE-0.29--0.67%20(10%2F10%20dams%20%3E%20persistence)-c4552f)

## What this is

A research project forecasting **next-1-to-12-week inflow volumes** (P10/P50/P90 quantiles) for 10 Peninsular-Indian reservoirs — Almatti, Tungabhadra, Krishnaraja Sagara, Mettur, Nagarjuna Sagar, Srisailam, Jayakwadi, Ujjani, Sardar Sarovar, Ukai — using a **GAT (spatial) + TCN (temporal) + ENSO/IOD cross-attention** architecture, with a physical mass-balance storage stage.

## Headline results (held-out 2024, week-1, seed-averaged over 3 seeds, ERA5 true rainfall)

| Reservoir | GNN NSE | Persistence | Climatology |
|---|---|---|---|
| Ukai | 0.672 ± 0.047 | 0.650 | 0.791 |
| Tungabhadra | 0.642 ± 0.059 | 0.612 | 0.542 |
| Srisailam | 0.605 ± 0.051 | 0.418 | 0.453 |
| Mettur | 0.600 ± 0.031 | 0.352 | −2.376 |
| Krishnaraja Sagara | 0.586 ± 0.042 | 0.459 | 0.382 |
| Almatti | 0.582 ± 0.036 | 0.488 | 0.533 |
| Sardar Sarovar | 0.540 ± 0.040 | 0.376 | 0.530 |
| Ujjani | 0.497 ± 0.073 | 0.338 | 0.358 |
| Nagarjuna Sagar | 0.413 ± 0.063 | 0.162 | 0.526 |
| Jayakwadi | 0.291 ± 0.035 | −0.318 | 0.369 |

**10/10 reservoirs positive NSE · 10/10 beat persistence · 7/10 beat seasonal climatology · mean 0.543 · seed std ≤ 0.073.**

**Ablation (same model, only the rainfall feature changed):** true ERA5 precipitation vs surface-runoff proxy improves **10/10 reservoirs** (+0.033 mean NSE; Jayakwadi +0.062, Srisailam +0.051, KRS +0.045). Blending GNN with seasonal climatology keeps weeks 3–5 competitive (see `scripts/blend_eval.py`).

## LEVEL forecasting (physics-informed mass balance, `predict_releases` era)

Two complementary LEVEL results, both on held-out 2024 (storage NSE, TMC):

**1. Operational mode** (`scripts/eval_levels_v3.py` — GNN inflow + known releases, the standard reservoir-study assumption): week-1 mean **0.545** across 10 dams; **0.675 during the 2023 El Niño onset**.

**2. Physics-constrained model** (`src/models/physics_constrained_gnn.py` — differentiable mass-balance rollout inside the forward pass, no known-releases assumption): approaches persistence on most dams (NS −1.10 vs −1.13; KRS −1.07 vs −1.13) and beats the ΔS regressor on 7–8/10 (Tungabhadra −1.15 vs −65; NS −1.09 vs −35).

**Honest negative result** (`docs/levels_ds_negative_result.md`): a direct ΔS regressor (GradientBoosting) loses to storage persistence on 9/10 dams — weekly storage is persistence-dominated; the physics constraint prevents catastrophic drift but does not manufacture skill the features don't carry. Full details: `docs/levels_ds_negative_result.md`.


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
