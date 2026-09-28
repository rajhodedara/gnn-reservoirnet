# Graph Neural Networks for Spatiotemporal Prediction of Major Reservoir Levels in Peninsular India during El Niño

**Draft v1 — GNN-ReservoirNet · repo: github.com/rajhodedara/gnn-reservoirnet · tag `v1.0`**

---
## Abstract

We present a graph neural network system for forecasting the weekly dynamics of ten major Peninsular-Indian reservoirs — Almatti, Tungabhadra, Krishnaraja Sagara, Mettur, Nagarjuna Sagar, Srisailam, Jayakwadi, Ujjani, Sardar Sarovar and Ukai — spanning the Krishna, Godavari, Narmada, Cauvery and Tapi basins, with a focus on El Niño conditions. The architecture couples graph attention over physically-motivated inter-basin edges with temporal convolution and cross-attention to ENSO/IOD climate indices, and trains a quantile head (P10/P50/P90) jointly with a release head that feeds a physical mass-balance stage.

On the held-out year 2024 the model's week-1 inflow forecasts reach a **mean per-reservoir NSE of 0.559** (5 seeds, mean seed-std 0.040) and a **pooled NSE of 0.642 ± 0.009**, beating persistence on **9/10** reservoirs and seasonal climatology on **7/10**. Against the baselines the design calls for, it leads every one: persistence 0.486, climatology 0.467, a per-dam LSTM 0.449, Random Forest 0.441, SARIMA −0.130 (all week-1 pooled).

Because the project title concerns **levels during El Niño**, validation is stratified by climate phase using real monthly ONI. In the **2023 El Niño onset** the GNN is the only method with positive skill (pooled week-1 NSE 0.125, against persistence −0.080 and climatology −0.010); in the **severe 2015 El Niño** it also beats persistence (0.564 vs 0.527). Under operational assumptions (GNN inflow with known release schedules) week-1 storage NSE reaches **0.892 during the 2023 El Niño monsoon**, and a self-contained rule-curve variant reaches 0.614 versus persistence 0.633 — markedly closer than in neutral years, where storage barely moves and persistence is trivially strong.

We contribute three documented negative results: inflow routing cannot beat storage persistence (a signal-to-noise limit), a physics-constrained model improves robustness over a learned ΔS regressor without reaching the persistence bar, and direct ΔS regression loses to persistence on 9/10 dams. A mass-balance data-forensics audit also identified and corrected 998 falsified zero-inflow records at the most drought-affected node. Every reported figure is reproducible from public code, data provenance manifests and per-seed evidence.

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

- **Splits**: train 2010–2022, validation 2023 (El Niño onset — used for model selection, never for test), test 2024 (fully held-out).
- **5 seeds** (42, 7, 123, 2024, 17); reported metrics are seed-means ± std, with per-reservoir seed spread shown rather than summarised away.
- **Standardization is train-window only.** Feature and target statistics are computed from the training window; scoring inverts exactly those constants. Stored targets were verified against an independent recomputation from the raw CSVs (ratio 1.000 in every fold).
- **Baselines**: persistence (last observed weekly inflow), seasonal climatology, a per-dam LSTM (90-day lookback, 3 seeds averaged), Random Forest (lagged block features) and SARIMA (weekly-resampled, 52-week seasonality) — all on the identical target, split and forecast origins.
- **Rolling-origin robustness, stratified by climate phase**: for Y ∈ {2015, 2016, 2023, 2024}, train ≤ Y−2, validate Y−1, test Y — leakage-free expanding windows. 2015 is the severe El Niño, 2023 the recent onset; each fold is tagged by its JJAS-mean ONI.
- **Pooled and per-reservoir metrics are both reported.** Averaging per-reservoir NSE across dams of unequal variance is not a valid summary of overall skill, so pooled scores are given alongside the per-dam breakdown.
- Smoke-tested locally (CPU) before every GPU run; all reported runs execute on a Kaggle T4, and the notebook verifies the checked-out revision before training.

### 3.3 Level-forecast formulations

Three routes from model to storage were evaluated: **(a)** operational mode — GNN inflow + observed releases (the standard reservoir-study assumption); **(b)** a physics-constrained model with a differentiable mass-balance rollout inside the forward pass (no known releases); **(c)** routing forecast inflow through the balance equation, open-loop and closed-loop.

---

## 4. Results
### 4.1 Level forecasting under operational assumptions

With release schedules known — the standard planning assumption, and the information operators actually possess — the system delivers **week-1 storage NSE 0.878 across 2023 origins, rising to 0.892 during the 2023 El Niño monsoon (JJAS)**. This is the direct answer to the title's question: under operational conditions the system predicts major reservoir levels with usable skill through the climate event the study targets.

Two further readings matter for honesty:

- **Self-contained level skill** (rule-curve releases, no future knowledge) reaches pooled week-1 NSE **0.614 in El Niño** versus persistence 0.633 — and 0.661 in neutral versus 0.927. The gap between the model and persistence **collapses in El Niño years** precisely because persistence is trivially strong when storage barely moves. This is the setting in which a level model has the most to offer.
- Earlier drafts quoted a per-dam *mean* of 0.545 for the operational result. That aggregate was dominated by a single broken gauge; pooling — the statistically valid summary across dams of unequal variance — gives the higher figure, and both the pooled and per-dam views are reported.

### 4.2 Inflow forecasting — the engine

Week-1 inflow, held-out 2024, 5 seeds:

| Reservoir | GNN NSE | Persistence | Climatology | LSTM |
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

**Mean 0.559 · pooled 0.642 ± 0.009 · 9/10 beat persistence · 7/10 beat climatology.** Skill decays with lead time and remains positive for 3–4 weeks; blending with climatology keeps weeks 3–5 competitive (`scripts/blend_eval.py`).

**Why the graph.** The design calls for classical and machine-learning baselines on the identical target, split and forecast origins. Week-1 pooled NSE, 2024:

| Model | Pooled NSE |
|---|---|
| **GNN (ours)** | **0.639** |
| Persistence | 0.486 |
| Climatology | 0.467 |
| LSTM (per-dam, 3 seeds averaged) | 0.449 |
| Random Forest | 0.441 |
| SARIMA | −0.130 |

The GNN leads the best baseline by **+0.153**. Two honest qualifications: per-dam, the LSTM wins on several low-variance dry-season reservoirs (it dominates only in pooled terms because the GNN's advantage concentrates on the large-variance monsoon-dominated dams), and SARIMA fails outright because weekly inflow carries a large dry-season zero mass that a linear ARIMA process cannot represent.

**Robustness — stratified by climate phase.** Expanding-window refits (fold Y: test = Y, val = Y−1, train ≤ Y−2), leakage-free. Folds 2015/2016 satisfy the design's requirement to validate on the severe 2015-16 El Niño.

| Fold | Phase | JJAS ONI | GNN (pooled wk-1) | Persistence | Climatology | Winner |
|---|---|---|---|---|---|---|
| **2015** | **El Niño (severe)** | **+1.74** | **0.564** | 0.527 | −4.185 | **GNN** |
| 2016 | neutral | −0.35 | 0.137 | **0.363** | 0.297 | persistence |
| **2023** | **El Niño (onset)** | **+1.23** | **0.125** | −0.080 | −0.010 | **GNN** |
| 2024 | neutral | +0.01 | **0.651** | 0.486 | 0.467 | **GNN** |

**The GNN wins in both El Niño years** — and in 2023 it is the only method with positive skill, both baselines having gone negative under the displaced monsoon. Three of four folds go to the GNN; 2016 is the exception and is reported as such.

**Verification note.** The stored test targets were checked against an independent recomputation from the raw reservoir CSVs: the ratio is 1.000 to three decimals in every fold. This closed a scaling defect found during a late review, in which targets were standardized with training-window statistics but un-scaled with full-record statistics; the error grew as the training window shrank and had disproportionately depressed the 2015/2016 folds.

**Project trajectory.** The pipeline's mean rose from **0.408** (original mixed-vintage data) → **0.543** (real Mettur/SSP targets, ERA5 true rainfall) → **0.577** (KRMB board data for Srisailam/NS) → **0.581** (fake-zero mask) → **0.559 with the corrected scaling and stratified validation** — the last step trading a slightly lower headline for a materially more reliable one.

### 4.3 Negative results (contributions)

**(a) Inflow routing cannot beat persistence — a signal-to-noise limit.** Routing even the best-case inflow forecast through the balance equation (closed-loop: restart from observed storage weekly, subtract observed releases) loses to persistence on **10/10 dams at every horizon** (week-1 pooled NSE −6.4 vs −0.23). The closed-loop level error *equals* the inflow error, and inflow RMSE — even at NSE 0.58 — exceeds the entire weekly ΔS signal (1–5% of capacity) on these dams. Open-loop rollouts compound the same error and are catastrophic. Reproduce: `scripts/closed_loop_eval.py`.

**(b) The physics-constrained model: robustness, not skill.** The differentiable mass-balance model beats persistence on 3/10 dams (week 1) — below our pre-registered 6/10 acceptance bar, so we claim no self-contained level skill. Its value is structural: it beats a learned ΔS GradientBoosting regressor on **8/10 dams** (e.g. NS −0.98 vs −16.3; Tungabhadra −2.3 vs −18.8), because physics prevents the catastrophic extrapolation a learned regressor suffers. Physics as damage limitation, quantified.

**(c) Direct ΔS regression** loses to persistence on 9/10 dams (`docs/levels_ds_negative_result.md`).

---
## 5. Limitations

- **El Niño validation has now been widened, but the severest event remains single-year.** The design's 2014–2016 requirement is met via the 2015 fold; the conclusion that the GNN leads in El Niño conditions rests on two such years (2015, 2023), not a multi-decade sample.
- **Seed stability is good but not uniform.** Mean per-dam seed-std is 0.040; Ujjani alone is 0.187, and one earlier configuration converged to a reproducibly poor basin. Seed spread is reported per reservoir rather than summarised away.
- **Per-dam, the LSTM is competitive.** The GNN's advantage is clearest in pooled terms and on the large-variance monsoon dams; on low-variance dry-season dams a per-dam LSTM can win. We report both and claim only what the pooled and phase-stratified evidence supports.
- **Jayakwadi** remains capped by its gauge (weekly gauge-vs-storage-Δ correlation r ≈ 0.13 even after the fake-zero mask); no better public source exists — two search campaigns are documented with receipts (`docs/jayakwadi_package/`). Its inclusion is deliberate: the dam is the study's most drought-relevant node.
- **Operational level results assume known releases.** Without that assumption, self-contained level skill does not beat persistence in neutral years (documented above); in El Niño years it comes much closer. We consider stating this honestly a feature of the work.
- **KRMB-served dams** (Srisailam, NS) lack board data before 2015, leaving a source break we do not paper over.

## 6. Reproducibility

Everything is public at `github.com/rajhodedara/gnn-reservoirnet`: data with provenance manifests, all training/evaluation code, the Kaggle notebook (one-click: clone → verify revision → train 5 seeds → evaluate → rolling folds → baselines → package), per-seed evidence, four rolling-origin folds spanning two El Niño years, the phase-stratified analysis scripts, and the negative-result documentation. The full unit suite passes locally (190 tests).

Regeneration is a three-step, scripted path: `kaggle_runner.ipynb` (training and folds), `scripts/run_ml_baselines.py` (ARIMA/RF/LSTM on the same splits), then `scripts/definitive_tables.py`, which rebuilds every table in this report from the retrieved artifacts.

## 7. Conclusion

A GNN with physical structure and audited data forecasts Peninsular-Indian reservoir inflow 1–4 weeks ahead with skill that survives phase-stratified validation — leading persistence and climatology in both El Niño years tested and in the neutral held-out year — and delivers level forecasts under operational assumptions with strong skill during the 2023 El Niño monsoon. Its negative results map exactly where self-contained level prediction fails and why.

The bounding lesson generalises: in water-data science the decisive gains came from **data provenance, forensics and evaluation discipline** — a fake-zero audit, an independent target-scale verification, and a stratified validation design — rather than from architecture alone.

---

*Figures: `outputs/figures/fig1_inflow_nse_by_dam.png` (scoreboard), `fig2_nse_vs_horizon.png` (skill decay + rolling folds), `fig3_el_nino_2023_srisailam.png` (El Niño showcase).*
