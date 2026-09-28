# GNN-ReservoirNet

Spatio-temporal GNN forecasting weekly inflow volumes for **10 major dams of Peninsular India** - built on a fully provenance-tracked, real-measured dataset, benchmarked against persistence and climatology baselines.

![status](https://img.shields.io/badge/dataset-wris__v2--verified-2b5f75) ![results](https://img.shields.io/badge/held--out%20NSE-0.29--0.67%20(10%2F10%20dams%20%3E%20persistence)-c4552f)

## What this is

A research project forecasting **next-1-to-12-week inflow volumes** (P10/P50/P90 quantiles) for 10 Peninsular-Indian reservoirs - Almatti, Tungabhadra, Krishnaraja Sagara, Mettur, Nagarjuna Sagar, Srisailam, Jayakwadi, Ujjani, Sardar Sarovar, Ukai - using a **GAT (spatial) + TCN (temporal) + ENSO/IOD cross-attention** architecture, with a physical mass-balance storage stage.

## Headline results (held-out 2024, week-1, 5 seeds, ERA5 true rainfall)

| Reservoir | GNN NSE (5 seeds) | Persistence | Climatology | LSTM (3 seeds) |
|---|---|---|---|---|
| Ukai | 0.644 ± 0.030 | 0.650 | 0.791 | 0.695 |
| Sardar Sarovar | 0.626 ± 0.031 | 0.376 | 0.686 | 0.706 |
| Almatti | 0.624 ± 0.032 | 0.488 | 0.533 | 0.419 |
| Tungabhadra | 0.621 ± 0.021 | 0.612 | 0.542 | 0.669 |
| Nagarjuna Sagar | 0.601 ± 0.016 | 0.220 | 0.292 | 0.376 |
| Srisailam | 0.589 ± 0.013 | 0.498 | 0.350 | 0.291 |
| Krishnaraja Sagara | 0.582 ± 0.024 | 0.459 | 0.382 | 0.523 |
| Mettur | 0.565 ± 0.012 | 0.352 | 0.354 | 0.577 |
| Ujjani | 0.460 ± 0.187 | 0.338 | 0.357 | 0.175 |
| Jayakwadi | 0.276 ± 0.029 | 0.011 | 0.351 | 0.366 |

**10/10 reservoirs positive NSE · 9/10 beat persistence (not Ukai) · 7/10 beat seasonal climatology · mean 0.559 · mean seed-std 0.040.**

**Pooled across all dams and origins (the statistically valid summary), the GNN scores NSE 0.642 ± 0.009** — every seed between 0.629 and 0.654.

**Provenance note.** These results come from the corrected pipeline (train-window-only standardization; commit `0fbbc1b`). The stored test targets were independently verified against a raw recomputation from `data/raw/wris_v2` — the ratio is 1.00 to three decimals for every fold, so the un-scaling is exact.

## Baselines — is the graph actually needed?

Same target, same split, same origins (week-1 pooled NSE, 2024):

| Model | Pooled NSE |
|---|---|
| **GNN (ours)** | **0.639** |
| Persistence | 0.486 |
| Climatology | 0.467 |
| LSTM (per-dam, 3 seeds averaged) | 0.449 |
| Random Forest | 0.441 |
| SARIMA | −0.130 |

The GNN beats the best baseline by **+0.153**. Per-dam against the LSTM it is closer (the LSTM wins on several low-variance dry-season dams), so both views are reported: pooled favours the GNN clearly, per-dam is mixed. SARIMA fails on these series because weekly inflow has a large dry-season zero mass that a linear ARIMA process cannot represent.

## LEVEL forecasting (physics-informed mass balance, `predict_releases` era)

Two complementary LEVEL results, both on held-out 2024 (storage NSE, TMC):

**1. Operational mode** (GNN inflow + *actual* releases — the standard reservoir-study assumption): week-1 storage NSE **0.878** across 2023 origins, **0.892** during the 2023 El Niño monsoon (JJAS). Self-contained variant (rule-curve releases, no future knowledge): pooled **0.614 in El Niño** vs persistence 0.633, and 0.661 in neutral vs 0.927 — i.e. the level model is markedly more competitive in El Niño years, because persistence is trivially strong when storage barely moves.

**2. Physics-constrained model** (`src/models/physics_constrained_gnn.py` - differentiable mass-balance rollout inside the forward pass, no known-releases assumption; final GPU run, per-dam training): beats persistence on **3/10 dams at week 1** (Srisailam **0.924 vs 0.908**, NS, Ukai), 3/10 at week 4, 4/10 at week 12 - below our 6/10 acceptance bar, so we do **not** claim persistence-beating self-contained level skill. Its real value is robustness: it beats the direct ΔS GradientBoosting regressor on **8/10 dams at week 1** (NS -0.98 vs -16.3; Tungabhadra -2.3 vs -18.8; KRS -3.3 vs -12.8) - the differentiable mass-balance structure prevents the catastrophic failures a learned ΔS regressor suffers.

**Fundamental negative result - inflow routing cannot beat persistence** (run #8 closed-loop test): routing even the *best-case* GNN inflow forecast through mass balance (restart from observed storage each week, subtract observed releases) loses to persistence on **10/10 dams at every horizon** (week-1 pooled NSE **-6.4 vs -0.23**). Reason: the closed-loop level error equals the inflow forecast error, and inflow RMSE (even at NSE 0.58) is larger than the entire weekly ΔS signal (1-5% of capacity) on these dams. Open-loop 12-week rollouts (release-head routed) compound the same error and are catastrophic (NSE -7 to -103). Level skill must come from *predicting ΔS directly* (physics model, above) or from known releases (operational mode, above) - not from inflow routing. Reproduce: `scripts/closed_loop_eval.py`.

**Honest negative result** (`docs/levels_ds_negative_result.md`): a direct ΔS regressor (GradientBoosting) loses to storage persistence on 9/10 dams - weekly storage is persistence-dominated; the physics constraint prevents catastrophic drift but does not manufacture skill the features don't carry. Full details: `docs/levels_ds_negative_result.md`.


## Robustness: rolling-origin across climate phases

Expanding-window refits (fold Y: test = Y, val = Y-1, train ≤ Y-2), leakage-free. Folds 2015/2016 satisfy the design's requirement to validate on the **severe 2015-16 El Niño**.

| Fold | Phase | JJAS ONI | GNN (pooled wk-1) | Persistence | Climatology | Winner |
|---|---|---|---|---|---|---|
| **2015** | **El Niño (severe)** | **+1.74** | **0.564** | 0.527 | −4.185 | **GNN** |
| 2016 | neutral | −0.35 | 0.137 | **0.363** | 0.297 | persistence |
| **2023** | **El Niño (onset)** | **+1.23** | **0.125** | −0.080 | −0.010 | **GNN** |
| 2024 | neutral | +0.01 | **0.651** | 0.486 | 0.467 | **GNN** |

**In both El Niño years the GNN wins** — and in 2023 it is the only method with positive skill (both baselines go negative). In the severe 2015 event it also beats persistence outright. 2016 is the one fold where persistence wins; three of four folds go to the GNN.

Caveat: 2015 includes a near-zero-variance dam (drought), excluded from the pooled figure and named explicitly by `scripts/definitive_tables.py`.

## The dataset (`data/raw/wris_v2/`)

Ten reservoir CSVs - daily, 2010-01-01 → 2024-12-31, 5,479 rows each:
`Date, Reservoir_Name, Inflow (m3/s), Storage (TMC)` - inflow is agency-gauge-derived where a gauge exists, but **for several reservoirs a large fraction of days are storage-derived** (the project's own QA gate reports `match_inflow_eq_maxDeltaS_pct` of 35-63% for almatti/jayakwadi/ujjani, with near-zero `corr_inflow_vs_nextday_storage_gain`; see `runs/data_qa_report.json`), so per-node provenance should be read from the manifests rather than assumed uniform:

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
- **Jayakwadi** capped by its gauge (Dhalegaon: 64.8% zeros, monsoon pulse magnitudes poorly tracked - weekly gauge-vs-storage-Δ correlation r≈0.13 even after our fake-zero mask; no better source found after exhaustive search, incl. two documented dead ends: `docs/jayakwadi_package/OCR_DEADEND.md`)
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
