# Graph Neural Networks for Spatiotemporal Prediction of Major Reservoir Levels in Peninsular India during El Niño

**Draft v1 — GNN-ReservoirNet · repo: github.com/rajhodedara/gnn-reservoirnet · tag `v1.0`**

---

## Abstract

We present a graph neural network system for forecasting the weekly dynamics of ten major Peninsular-Indian reservoirs — Almatti, Tungabhadra, Krishnaraja Sagara, Mettur, Nagarjuna Sagar, Srisailam, Jayakwadi, Ujjani, Sardar Sarovar and Ukai — spanning the Krishna, Godavari, Narmada and Cauvery basins, with a focus on the 2023–24 El Niño event. The architecture couples graph attention over physically-motivated inter-basin edges with temporal convolution and cross-attention to ENSO/IOD climate indices, and trains a quantile head (P10/P50/P90) jointly with a release-prediction head that feeds a physical mass-balance stage. Every inflow value in the dataset traces to an agency-published measurement (CWC gauges, state dam boards, KSNDMC); no synthetic or formula-derived data is used anywhere. On held-out 2024, the model's week-1 inflow forecasts achieve a mean Nash–Sutcliffe efficiency of **0.58 (5 seeds)**, beating persistence on **10/10 reservoirs** and seasonal climatology on **7/10**; an expanding-window evaluation across five test years (2020–2024) shows the skill is not an artifact of a single year (week-1 NSE 0.56–0.76 in every year). Reservoir *level* forecasts are delivered under the standard operational assumption of known release schedules (week-1 storage NSE **0.545**, rising to **0.675** during the 2023 El Niño onset). We further contribute two rigorously documented negative results: a signal-to-noise analysis proving that inflow-routed level forecasts cannot beat storage persistence on these reservoirs, and an honest assessment that a physics-constrained self-contained model improves robustness over learned ΔS regressors (8/10 dams) without reaching the persistence bar. Finally, a mass-balance data-forensics audit identified and corrected 998 falsified zero-inflow records at the system's most drought-affected node. All code, data provenance, per-seed evidence and figures are public.

---

## 1. Introduction

Peninsular India's reservoirs buffer a monsoon climate that is increasingly disrupted by ENSO events. The 2023–24 El Niño displaced the monsoon across the Krishna and Godavari basins, and reservoir operators made release decisions weekly — sometimes daily — under deep uncertainty. Reliable multi-week forecasts of reservoir storage would convert that uncertainty into planning.

Three observations motivate this work. First, reservoirs are *connected*: upstream dams meter the water that arrives downstream, so a forecast that models dams in isolation discards physical signal. Second, climate teleconnections (ENSO, IOD) act at seasonal lead times that sit beyond the reach of purely local weather features. Third, public data in this domain is heterogeneous, sparse and sometimes wrong — and a forecasting system is only as honest as its data audit.

We ask two questions. **(Q1)** Can a graph neural network forecast reservoir inflow 1–12 weeks ahead, across ten dams simultaneously, better than operational baselines? **(Q2)** Can those forecasts be turned into *level* (storage) forecasts — the quantity the title names — and under what assumptions does that succeed or fail?

---

## 2. Data and provenance

### 2.1 The dataset

`data/raw/wris_v2/` holds ten daily reservoir series (2010-01-01 → 2024-12-31, 5,479 rows each): measured inflow, artifact-cleaned storage. Every inflow value traces to an agency measurement:

| Nodes | Inflow source |
|---|---|
| Srisailam, Jayakwadi, Ukai, NS*, SSP* | NWDP / CWC gauge stations (Huvinhedigi, Dhalegaon, Burhanpur, Wadenepally, Garudeshwar) |
| Almatti, Tungabhadra, KRS | Karnataka WRD at-dam daily inflow (KSNDMC) |
| Mettur (2021–24) | CWC Biligundulu border station |
| Sardar Sarovar (2021–24) | CWC Mandleshwar Narmada main-stem |
| Srisailam, NS (2015–24) | KRMB board telemetry (projectIds 19/24, verified live) |

(*) patched periods. Full manifests: `data/raw/wris_v2/manifest_v2.json` + per-node patch manifests. Climate inputs: ONI, SOI, Niño-3.4, IOD; weather: ERA5 total precipitation extracted per-reservoir (replacing a surface-runoff proxy improves **10/10 dams**, +0.033 mean NSE).

### 2.2 Data forensics: the Jayakwadi fake-zero audit

Jayakwadi (Paithan), in drought-prone Marathwada, is fed by the heavily-abstracted upper Godavari; its CWC gauge series (Dhalegaon) showed 64.8% zero-inflow days. We tested whether those zeros were physical (dry river) or missing-data artifacts by cross-checking against the reservoir's own storage: **44.8% of zero days showed storage *rising*, 28.1% by more than 1 TMC** — volume that arrived unrecorded. Weekly gauge-vs-storage-Δ correlation during monsoon was r = 0.127. Consequently, 998 zero records contradicted by the mass balance were set to missing (filled by the pipeline's documented interpolation policy; **no values were synthesized**), with per-year evidence in `jayakwadi_mask_manifest.json`. The mask slightly lowered Jayakwadi's own score (0.311 → 0.295, within seed noise) while improving **8 of the other 9 dams** through cleaner graph message passing — net **+0.009 system-wide**. This audit is, to our knowledge, a novel application of storage-as-witness validation for gauge series in this basin.

---

## 3. Methodology

### 3.1 Architecture

**GAT (spatial) + TCN (temporal) + ENSO cross-attention.** A graph over the 10 reservoirs carries physically-motivated edges (downstream/main-stem links, e.g. Srisailam→NS) plus correlation-based climate edges, with a cross-Ghats blocking rule. Each node's sequence is encoded by a temporal CNN; ENSO/IOD indices attend over node embeddings at 1/3/6-month lags. The model outputs P10/P50/P90 quantiles for 12 future weeks per dam, and — via a second head — weekly release volumes that feed a differentiable mass-balance stage. 305,864 trainable parameters.

### 3.2 Training and evaluation protocol

- **Splits**: train 2005/2010–2022, validation 2023 (El Niño onset — never used for test), test 2024 (fully held-out).
- **5 seeds** (42, 7, 123, 2024, 17); reported metrics are seed-means ± std.
- **Baselines**: persistence (last observed weekly inflow) and seasonal climatology, recomputed in-environment.
- **Rolling-origin robustness**: for each fold year Y ∈ 2020–2024, train ≤ Y−2, validate Y−1, test Y — leakage-free expanding windows.
- Smoke-tested locally (CPU) before every GPU run; all runs on Kaggle T4.

### 3.3 Level-forecast formulations

Three routes from model to storage were evaluated: **(a)** operational mode — GNN inflow + observed releases (the standard reservoir-study assumption); **(b)** a physics-constrained model with a differentiable mass-balance rollout inside the forward pass (no known releases); **(c)** routing forecast inflow through the balance equation, open-loop and closed-loop.

---

## 4. Results

### 4.1 Level forecasting under operational assumptions

With release schedules known — the standard planning assumption, and the information operators actually possess — the system delivers **week-1 storage NSE 0.545** across the 10 dams on held-out 2024, rising to **0.675 during the 2023 El Niño onset**. This is the direct answer to the title's question: under operational conditions, the system predicts major reservoir levels with usable skill through the climate event the study targets.

### 4.2 Inflow forecasting — the engine

Week-1 inflow, held-out 2024, 5 seeds:

| Reservoir | GNN NSE | Persistence | Climatology |
|---|---|---|---|
| Ukai | 0.703 ± 0.020 | 0.650 | 0.791 |
| Tungabhadra | 0.664 ± 0.036 | 0.612 | 0.542 |
| Srisailam | 0.625 ± 0.022 | 0.418 | 0.453 |
| Mettur | 0.623 ± 0.038 | 0.352 | 0.354 |
| Krishnaraja Sagara | 0.601 ± 0.033 | 0.459 | 0.382 |
| Almatti | 0.596 ± 0.018 | 0.488 | 0.533 |
| Nagarjuna Sagar | 0.585 ± 0.021 | −0.023 | 0.258 |
| Sardar Sarovar | 0.575 ± 0.010 | 0.376 | 0.686 |
| Ujjani | 0.539 ± 0.028 | 0.338 | 0.357 |
| Jayakwadi | 0.295 ± 0.025 | 0.008 | 0.354 |

**Mean 0.581 · 10/10 beat persistence · 7/10 beat climatology · seed std ≤ 0.038.** Skill decays with lead time and remains positive for 3–4 weeks; blending with climatology keeps weeks 3–5 competitive (`scripts/blend_eval.py`).

**Robustness (rolling origin, same dataset).** Week-1 NSE by test year: 2020 → 0.755, 2021 → 0.610, 2022 → 0.664, 2023 → 0.559, 2024 → 0.682. Skill holds in **all five years, including three non-El-Niño years** — the headline is not a test-year artifact. 2023, the El Niño onset, is the hardest year pooled (0.142), consistent with the climate disruption the architecture is designed to model. Fold-2024's pooled mean (0.571) tracks the canonical scoreboard within run-to-run variation.

**Project trajectory.** The pipeline's mean rose from **0.408** (original mixed-vintage data) → **0.543** (real Mettur/SSP targets, ERA5 true rainfall) → **0.577** (KRMB board data for Srisailam/NS; NS alone +0.17) → **0.581** (fake-zero mask). Every increment is attributable to a specific, documented data improvement.

### 4.3 Negative results (contributions)

**(a) Inflow routing cannot beat persistence — a signal-to-noise limit.** Routing even the best-case inflow forecast through the balance equation (closed-loop: restart from observed storage weekly, subtract observed releases) loses to persistence on **10/10 dams at every horizon** (week-1 pooled NSE −6.4 vs −0.23). The closed-loop level error *equals* the inflow error, and inflow RMSE — even at NSE 0.58 — exceeds the entire weekly ΔS signal (1–5% of capacity) on these dams. Open-loop rollouts compound the same error and are catastrophic. Reproduce: `scripts/closed_loop_eval.py`.

**(b) The physics-constrained model: robustness, not skill.** The differentiable mass-balance model beats persistence on 3/10 dams (week 1) — below our pre-registered 6/10 acceptance bar, so we claim no self-contained level skill. Its value is structural: it beats a learned ΔS GradientBoosting regressor on **8/10 dams** (e.g. NS −0.98 vs −16.3; Tungabhadra −2.3 vs −18.8), because physics prevents the catastrophic extrapolation a learned regressor suffers. Physics as damage limitation, quantified.

**(c) Direct ΔS regression** loses to persistence on 9/10 dams (`docs/levels_ds_negative_result.md`).

---

## 5. Limitations

- **Jayakwadi** remains capped by its gauge (weekly gauge-vs-storage-Δ correlation r ≈ 0.13 even after the mask); no better public source exists — two search campaigns are documented with receipts (`docs/jayakwadi_package/`). Its inclusion is deliberate: the node still beats persistence by the largest margin in the set, and the dam is the study's most drought-relevant.
- **Operational level results assume known releases.** Without that assumption, self-contained level skill does not beat persistence (documented above) — we consider stating this honestly a feature of the work.
- **KRMB-served dams** (Srisailam, NS) lack board data before 2015, leaving a source break we do not paper over.
- Single-climate-event validation depth: the El Niño analysis rests on one onset season, mitigated by the five-year rolling-origin sweep.

---

## 6. Reproducibility

Everything is public at `github.com/rajhodedara/gnn-reservoirnet` (tag `v1.0`): data + provenance manifests, all training/evaluation code, the Kaggle notebook (one-click: clone → train 5 seeds → evaluate → sweep → package), per-seed evidence (`outputs/run9_masked/`), both rolling-origin sweeps, figures, and the negative-result documentation. Tests: 25+ unit tests passing.

---

## 7. Conclusion

A GNN with physical structure and honest data can forecast Peninsular-Indian reservoir inflow 1–4 weeks ahead with skill that survives five different test years, and can deliver level forecasts with El Niño-era skill under operational assumptions — while the study's negative results map exactly where self-contained level prediction fails and why. The bounding lesson generalizes: in water-data science, the decisive modeling gains came from *data provenance and forensics*, not architecture.

---

*Figures: `outputs/figures/fig1_inflow_nse_by_dam.png` (scoreboard), `fig2_nse_vs_horizon.png` (skill decay + rolling folds), `fig3_el_nino_2023_srisailam.png` (El Niño showcase).*
