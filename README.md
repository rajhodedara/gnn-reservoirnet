# GNN-ReservoirNet

Spatio-temporal GNN forecasting weekly inflow volumes for **10 major dams of Peninsular India** - built on a fully provenance-tracked, real-measured dataset, benchmarked against persistence and climatology baselines.

![status](https://img.shields.io/badge/dataset-wris__v2--verified-2b5f75) ![results](https://img.shields.io/badge/held--out%20NSE-0.29--0.67%20(10%2F10%20dams%20%3E%20persistence)-c4552f)

## What this is

A research project forecasting **next-1-to-12-week inflow volumes** (P10/P50/P90 quantiles) for 10 Peninsular-Indian reservoirs - Almatti, Tungabhadra, Krishnaraja Sagara, Mettur, Nagarjuna Sagar, Srisailam, Jayakwadi, Ujjani, Sardar Sarovar, Ukai - using a **GAT (spatial) + TCN (temporal) + ENSO/IOD cross-attention** architecture, with a physical mass-balance storage stage.

## Headline results (held-out 2024, week-1, seed-averaged over 5 seeds, ERA5 true rainfall)

| Reservoir | GNN NSE | Persistence | Climatology |
|---|---|---|---|
| Ukai | 0.705 ± 0.019 | 0.650 | 0.791 |
| Tungabhadra | 0.644 ± 0.024 | 0.612 | 0.542 |
| Srisailam | 0.614 ± 0.018 | 0.418 | 0.453 |
| Mettur | 0.607 ± 0.007 | 0.352 | 0.354 |
| Almatti | 0.587 ± 0.021 | 0.488 | 0.533 |
| Krishnaraja Sagara | 0.585 ± 0.014 | 0.459 | 0.382 |
| Nagarjuna Sagar | 0.573 ± 0.018 | -0.023 | 0.258 |
| Sardar Sarovar | 0.569 ± 0.018 | 0.376 | 0.686 |
| Ujjani | 0.520 ± 0.017 | 0.338 | 0.357 |
| Jayakwadi | 0.311 ± 0.022 | 0.008 | 0.354 |

**10/10 reservoirs positive NSE · 10/10 beat persistence · 7/10 beat seasonal climatology · mean 0.572 · 5 seeds · seed std ≤ 0.024.** (Run #8: KRMB board data for NS + Srisailam, Mettur/SSP target patches, ERA5 true rainfall, release head trained jointly. NS gained +0.17 from its own real KRMB data - from the weakest node to a climatology-beater.)

**Ablation (same model, only the rainfall feature changed):** true ERA5 precipitation vs surface-runoff proxy improves **10/10 reservoirs** (+0.033 mean NSE; Jayakwadi +0.062, Srisailam +0.051, KRS +0.045). Blending GNN with seasonal climatology keeps weeks 3-5 competitive (see `scripts/blend_eval.py`).

## LEVEL forecasting (physics-informed mass balance, `predict_releases` era)

Two complementary LEVEL results, both on held-out 2024 (storage NSE, TMC):

**1. Operational mode** (`scripts/eval_levels_v3.py` - GNN inflow + known releases, the standard reservoir-study assumption): week-1 mean **0.545** across 10 dams; **0.675 during the 2023 El Niño onset**.

**2. Physics-constrained model** (`src/models/physics_constrained_gnn.py` - differentiable mass-balance rollout inside the forward pass, no known-releases assumption; final GPU run, per-dam training): beats persistence on **3/10 dams at week 1** (Srisailam **0.924 vs 0.908**, NS, Ukai), 3/10 at week 4, 4/10 at week 12 - below our 6/10 acceptance bar, so we do **not** claim persistence-beating self-contained level skill. Its real value is robustness: it beats the direct ΔS GradientBoosting regressor on **8/10 dams at week 1** (NS -0.98 vs -16.3; Tungabhadra -2.3 vs -18.8; KRS -3.3 vs -12.8) - the differentiable mass-balance structure prevents the catastrophic failures a learned ΔS regressor suffers.

**Fundamental negative result - inflow routing cannot beat persistence** (run #8 closed-loop test): routing even the *best-case* GNN inflow forecast through mass balance (restart from observed storage each week, subtract observed releases) loses to persistence on **10/10 dams at every horizon** (week-1 pooled NSE **-6.4 vs -0.23**). Reason: the closed-loop level error equals the inflow forecast error, and inflow RMSE (even at NSE 0.58) is larger than the entire weekly ΔS signal (1-5% of capacity) on these dams. Open-loop 12-week rollouts (release-head routed) compound the same error and are catastrophic (NSE -7 to -103). Level skill must come from *predicting ΔS directly* (physics model, above) or from known releases (operational mode, above) - not from inflow routing. Reproduce: `scripts/closed_loop_eval.py`.

**Honest negative result** (`docs/levels_ds_negative_result.md`): a direct ΔS regressor (GradientBoosting) loses to storage persistence on 9/10 dams - weekly storage is persistence-dominated; the physics constraint prevents catastrophic drift but does not manufacture skill the features don't carry. Full details: `docs/levels_ds_negative_result.md`.


## Robustness: rolling-origin across five test years (2020-2024)

Expanding-window refits (fold Y: test = Y, val = Y-1, train ≤ Y-2), 1 seed per fold, leakage-free - the sweep machinery lives in `scripts/rolling_origin_eval.py` + the notebook's Step 12:

| Test year | Week-1 NSE | Week-4 | Week-12 | Pooled mean |
|---|---|---|---|---|
| 2020 | 0.755 | 0.150 | -0.083 | 0.549 |
| 2021 | 0.602 | 0.551 | 0.344 | 0.356 |
| 2022 | 0.671 | 0.379 | -0.053 | 0.641 |
| 2023 (El Niño onset) | 0.556 | 0.149 | 0.141 | 0.156 |
| 2024 | 0.683 | 0.246 | -0.289 | 0.572 |

**Week-1 skill holds in all five years (0.56-0.75), including three non-El-Niño years - the headline result is not a test-year artifact.** Fold-2024's pooled mean (0.5716) reproduces the canonical 5-seed scoreboard (0.5715) as a consistency anchor. Week-12 skill is honestly year-dependent (positive 2021/2023, near-zero elsewhere); 2023's El Niño onset is the hardest year pooled.

## The dataset (`data/raw/wris_v2/`)

Ten reservoir CSVs - daily, 2010-01-01 → 2024-12-31, 5,479 rows each:
`Date, Reservoir_Name, Inflow (m3/s), Storage (TMC)` - **every inflow value is a real agency measurement**:

| Nodes | Inflow source | Provenance |
|---|---|---|
| Srisailam, Jayakwadi, Ukai, NS*, SSP* | NWDP CWC gauge stations (Huvinhedigi, Dhalegaon, Burhanpur, Wadenepally, Garudeshwar) | `manifest_v2.json` + per-node patch manifests |
| Almatti, Tungabhadra, Krishnaraja Sagara | Karnataka WRD at-dam daily inflow (KSNDMC) | `ka_inflow_patch_manifest.json` |
| Mettur (2021-24) | CWC **Biligundulu** border station | `mettur_target_patch_manifest.json` |
| Sardar Sarovar (2021-24) | CWC **Mandleshwar** Narmada main-stem | `sardar_sarovar_target_patch_manifest.json` |

Storage columns are artifact-cleaned (unit de-mixing, impossible-spike masking, dead-storage handling) - see `scripts/build_wris_v2.py` and the QA gate `scripts/qa_wris_data.py`. Jayakwadi's inflow additionally passed a **mass-balance fake-zero audit**: 998 zero-inflow days (28% of its zeros) were contradicted by same-day storage rises > 1 TMC and set to missing (filled by the documented interpolation policy, no synthesis) - evidence and per-year counts in `data/raw/wris_v2/jayakwadi_mask_manifest.json`, reproducible via `scripts/patch_jayakwadi_mask.py`.

\* NS uses the documented Huvinhedigi upstream-Krishna proxy (Wadenepally verification in git history); SSP 2010-2020 remains Garudeshwar (release-contaminated, flagged).

## Climate inputs

- ENSO: ONI, SOI, Niño3.4 + **IOD (DMI)** - daily, 2005-2026 (`data/raw/enso/combined_climate_indices.csv`)
- ERA5-Land at each dam: surface runoff, evaporation, soil moisture - precomputed point extraction (`data/raw/era5/reservoir_era5_daily.csv`)
- Cross-attention conditions the GNN on climate; ENSO stratification (El Niño vs neutral) evaluated per split

## Reproduce

**Locally** (no GPU needed for data work):
```bash
python scripts/build_wris_v2.py                 # rebuild dataset
python scripts/qa_wris_data.py --dir data/raw/wris_v2   # QA gate
python scripts/run_baselines.py                 # baselines
python -m pytest tests/ -q                      # 22 tests
```

**Training** (GPU - Kaggle notebook `kaggle_runner.ipynb`, clones this repo):
```bash
python main.py --config configs/default_config.yaml --seed 42
# evaluates val (2023) AND held-out test (2024), saves predictions_*.npz
python scripts/blend_eval.py --seed-dir seed42   # GNN + climatology blending
python scripts/diagnose_nodes.py                 # per-node predictability
```

## Honest limitations

- **Skill horizon**: weeks 1-2 standalone; **blending with climatology keeps weeks 3-5 competitive** (see `scripts/blend_eval.py`)
- **Jayakwadi** capped by its gauge (Dhalegaon: 64.8% zeros, monsoon pulse magnitudes poorly tracked — weekly gauge-vs-storage-Δ correlation r≈0.13 even after our fake-zero mask; no better source found after exhaustive search, incl. two documented dead ends: `docs/jayakwadi_package/OCR_DEADEND.md`)
- **Sardar Sarovar 2010-2020** remains release-contaminated (Garudeshwar); Mandleshwar covers 2021+
- **No `tp` rainfall band** in the ERA5 bundle - weather slot uses surface runoff (sro)
- NS inflow remains a documented proxy

## Repo map

```
main.py                 training + dual-split evaluation entry point
src/                    models (GAT+TCN+attention), data loaders, training, evaluation, explainability
configs/                default_config.yaml, reservoirs.yaml (topology, capacities)
scripts/                dataset builder, QA gate, baselines, blending, diagnostics, patch tools
tests/                  25 tests (data cleaning, un-scaling, baselines, IOD merge, ERA5 features)
data/raw/wris_v2/       the verified dataset + provenance manifests
outputs/                archived run metrics (run1...run6)
```

## Data provenance chain

Every transformation is logged and reversible:
`data/raw/wris/` (real measurements + patch manifests) → `scripts/build_wris_v2.py` (scale correction, artifact masking, zero-floor handling) → `data/raw/wris_v2/` (training-ready) → `main.py` (z-scoring, weekly targets) → evaluation (exact inverse un-scaling).

## Reservoir research packages

- 📄 **[Jayakwadi Dam (Nathsagar) research package](docs/jayakwadi_package/report.html)** - map, timeline, charts, ecology, impact, 24 cited sources (also: `docs/jayakwadi_package/data/` for the dataset files)
