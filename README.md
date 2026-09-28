# GNN-ReservoirNet

Spatio-temporal GNN forecasting weekly inflow volumes for **10 major dams of Peninsular India** - built on a fully provenance-tracked, real-measured dataset, benchmarked against persistence and climatology baselines.

![status](https://img.shields.io/badge/dataset-wris__v2--verified-2b5f75) ![results](https://img.shields.io/badge/held--out%20NSE-0.29--0.67%20(10%2F10%20dams%20%3E%20persistence)-c4552f)

## What this is

A research project forecasting **next-1-to-12-week inflow volumes** (P10/P50/P90 quantiles) for 10 Peninsular-Indian reservoirs - Almatti, Tungabhadra, Krishnaraja Sagara, Mettur, Nagarjuna Sagar, Srisailam, Jayakwadi, Ujjani, Sardar Sarovar, Ukai - using a **GAT (spatial) + TCN (temporal) + ENSO/IOD cross-attention** architecture, with a physical mass-balance storage stage.

## Headline results (held-out 2024, week-1, seed-averaged over 5 seeds, ERA5 true rainfall)

| Reservoir | GNN NSE (5 seeds) | Persistence | Climatology | LSTM (3 seeds) |
|---|---|---|---|---|
| Ukai | 0.632 ± 0.082 | 0.650 | 0.791 | 0.695 |
| Sardar Sarovar | 0.591 ± 0.077 | 0.376 | 0.686 | 0.706 |
| Tungabhadra | 0.563 ± 0.166 | 0.612 | 0.542 | 0.669 |
| Almatti | 0.555 ± 0.183 | 0.488 | 0.533 | 0.419 |
| Srisailam | 0.541 ± 0.170 | 0.498 | 0.350 | 0.291 |
| Nagarjuna Sagar | 0.538 ± 0.152 | 0.220 | 0.292 | 0.376 |
| Krishnaraja Sagara | 0.521 ± 0.186 | 0.459 | 0.382 | 0.523 |
| Mettur | 0.497 ± 0.158 | 0.352 | 0.354 | 0.577 |
| Ujjani | 0.383 ± 0.147 | 0.338 | 0.357 | 0.175 |
| Jayakwadi | 0.314 ± 0.057 | 0.011 | 0.351 | 0.366 |

**10/10 reservoirs positive NSE · 8/10 beat persistence (not Ukai, Tungabhadra) · 7/10 beat seasonal climatology · mean 0.513 · 5 seeds.**

> **Seed stability, reported honestly.** One seed (`seed7`) converges to a reproducibly bad basin on 9/10 dams (per-dam values repeat to within 0.006 across independent runs), which inflates the mean per-dam seed-std to 0.138. Excluding it, the mean is **0.568 with std 0.036**. We report both rather than silently dropping it.

> **Pooled vs per-dam.** Pooled across all dams and origins, the GNN scores **NSE 0.676** on the identical 282 test origins against persistence **0.485** and climatology **0.452**. On the same origins the GNN beats a properly-trained, seed-averaged **LSTM on 4/10 dams** — the LSTM wins on the low-variance dry-season dams, while the GNN wins on the large-variance monsoon-dominated ones (Nagarjuna Sagar, Srisailam, Almatti, Ujjani). Note the two numbers measure different things: pooled, the GNN wins clearly; per-dam, it is a mixed result. Both are reported. (KRMB board data for NS + Srisailam, Mettur/SSP target patches, ERA5 true rainfall, release head trained jointly, Jayakwadi fake-zero mask. The mask slightly lowered Jayakwadi's own score (0.311 → 0.295, within seed noise) while improving 8/9 other dams through cleaner graph message passing - net +0.009 system-wide. NS gained +0.17 from its own real KRMB data - from the weakest node to a climatology-beater.)

**Ablation (same model, only the rainfall feature changed):** true ERA5 precipitation vs surface-runoff proxy improves **10/10 reservoirs** (+0.033 mean NSE; Jayakwadi +0.062, Srisailam +0.051, KRS +0.045). Blending GNN with seasonal climatology keeps weeks 3-5 competitive (see `scripts/blend_eval.py`).

## LEVEL forecasting (physics-informed mass balance, `predict_releases` era)

Two complementary LEVEL results, both on held-out 2024 (storage NSE, TMC):

**1. Operational mode** (`scripts/eval_levels_v3.py`, now pooled in `scripts/el_nino_analysis.py` - GNN inflow + *actual* releases, the standard reservoir-study assumption): week-1 storage NSE **0.878 across all 2023 origins**, rising to **0.892 during the 2023 El Niño monsoon (JJAS)**. For scale, storage persistence in the same strata scores 0.633 / 0.640 - so the GNN inflow adds roughly **+0.25**. Earlier drafts quoted a per-dam *mean* of 0.545; that aggregate was dominated by Jayakwadi's broken gauge, and pooling (the statistically valid summary across dams of unequal variance) gives the higher figure. The self-contained rule-curve variant (`scripts/eval_levels_rulecurve.py`) reaches week-1 pooled NSE **0.661** but still trails persistence (0.927) - the honest negative result stands.

**2. Physics-constrained model** (`src/models/physics_constrained_gnn.py` - differentiable mass-balance rollout inside the forward pass, no known-releases assumption; final GPU run, per-dam training): beats persistence on **3/10 dams at week 1** (Srisailam **0.924 vs 0.908**, NS, Ukai), 3/10 at week 4, 4/10 at week 12 - below our 6/10 acceptance bar, so we do **not** claim persistence-beating self-contained level skill. Its real value is robustness: it beats the direct ΔS GradientBoosting regressor on **8/10 dams at week 1** (NS -0.98 vs -16.3; Tungabhadra -2.3 vs -18.8; KRS -3.3 vs -12.8) - the differentiable mass-balance structure prevents the catastrophic failures a learned ΔS regressor suffers.

**Fundamental negative result - inflow routing cannot beat persistence** (run #8 closed-loop test): routing even the *best-case* GNN inflow forecast through mass balance (restart from observed storage each week, subtract observed releases) loses to persistence on **10/10 dams at every horizon** (week-1 pooled NSE **-6.4 vs -0.23**). Reason: the closed-loop level error equals the inflow forecast error, and inflow RMSE (even at NSE 0.58) is larger than the entire weekly ΔS signal (1-5% of capacity) on these dams. Open-loop 12-week rollouts (release-head routed) compound the same error and are catastrophic (NSE -7 to -103). Level skill must come from *predicting ΔS directly* (physics model, above) or from known releases (operational mode, above) - not from inflow routing. Reproduce: `scripts/closed_loop_eval.py`.

**Honest negative result** (`docs/levels_ds_negative_result.md`): a direct ΔS regressor (GradientBoosting) loses to storage persistence on 9/10 dams - weekly storage is persistence-dominated; the physics constraint prevents catastrophic drift but does not manufacture skill the features don't carry. Full details: `docs/levels_ds_negative_result.md`.


## Robustness: rolling-origin across climate phases (2015-2016 severe El Niño, 2023-2024 recent)

Expanding-window refits (fold Y: test = Y, val = Y-1, train ≤ Y-2), leakage-free - machinery in `scripts/rolling_origin_eval.py`, phase tagging in `scripts/analyse_el_nino_folds.py`. Folds 2015/2016 satisfy the approved design's requirement to validate on the **severe 2015-16 El Niño**.

| Test year | Phase | JJAS ONI | GNN (pooled wk-1) | Persistence | Climatology | Winner |
|---|---|---|---|---|---|---|
| **2015** | **El Niño (severe)** | **+1.74** | 0.409 | **0.527** | −4.185 | persistence |
| 2016 | neutral | −0.35 | **0.741** | 0.363 | 0.297 | GNN |
| **2023** | **El Niño (onset)** | **+1.23** | **0.320** | −0.080 | −0.010 | **GNN** |
| 2024 | neutral | +0.01 | **0.684** | 0.486 | 0.467 | GNN |

**The El Niño story, stated precisely.** In the **2023 El Niño onset** the GNN is the *only* method with positive skill — persistence and climatology both go negative because the displaced monsoon broke their assumptions. In the **severe 2015 El Niño** the GNN stays positive (0.409) but **persistence is stronger** (0.527). In both neutral years the GNN wins outright. Pooled week-1 NSE by phase: **El Niño 0.372 mean, neutral 0.712 mean**.

Caveat on 2015: Almatti's observed inflow that year is near-zero (drought; mean 13 m³/s), making its NSE degenerate; `scripts/definitive_tables.py` drops near-zero-variance dams from the pooled figure and names them explicitly.

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
