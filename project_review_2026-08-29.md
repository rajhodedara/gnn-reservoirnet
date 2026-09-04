# GNN-ReservoirNet — Comprehensive Project Review

**Date:** 2026-08-29 · **Scope:** full audit of design, data pipeline, model, training, evaluation, repo hygiene
**Verdict in one line:** Excellent research design document, but the current implementation **does not yet test the hypothesis in the design doc** — the prediction target, the graph, the climate inputs, and the evaluation all deviate from the approved design, and the reported metrics (KGE down to −130) are artifacts of these deviations rather than model performance.

---

## 1. What the project is

A two-stage, physics-informed GNN for forecasting reservoir levels/inflows in Peninsular India during El Niño:

- **Stage 1:** GAT (spatial) + TCN (temporal) + ENSO×IOD cross-attention → P10/P50/P90 inflow for weeks +1..+12
- **Stage 2:** deterministic mass-balance `S(t+1)=S(t)+I(t)−E(t)−R(t)` → storage
- Design doc (`gnn_reservoir_design_final.md`) went through a 5-agent review and is genuinely strong.

**Codebase:** `main.py` (pipeline) + `src/` package + `tests/` + Streamlit dashboard + `kaggle_stage/` (Kaggle copy) + ~780 MB of zipped data in the repo root. Training evidently ran on Kaggle (`runs/best_model_finetune.pt`, TensorBoard event files).

---

## 2. What's good

| Area | Evidence |
|---|---|
| Design rigor | 16+ objections resolved, explicit decision log, honest scoping (drought tool, not flood forecasting) |
| Modular code | Clean `src/{data,models,training,evaluation,explainability}` separation; typed; documented |
| Right ML instincts | Direct multi-step head (no autoregressive drift), quantile loss, temporal split by year, early stopping, grad clipping, AMP |
| Engineering extras | ONNX export path, Integrated Gradients + attention maps, Streamlit dashboard, multi-source download script with manifest |
| Tests exist | 4 test files covering model/training/inference/graph (though only shape smoke tests) |

---

## 3. Critical weak points (ranked)

### 🔴 P0-1 — The prediction target is not inflow, and it contradicts the design's own Decision #1
- `data/raw/wris/manifest.json` proves the downloaded WRIS data had **only `[Date, storage]`** — no inflow.
- The current CSVs' `inflow` column matches `max(Δstorage, 0)` on 30–91% of days (corr ≈ 0.7–0.9); e.g. Srisailam 2015-07-10: storage 0.66→1.85, "inflow" = 1.19 exactly.
- Consequences:
  - The target is **net storage gain** — exactly what the design doc said is *not learnable* ("releases are governed by tribunal politics, not learnable physics"). Human releases contaminate the target.
  - It is **zero-inflated**: 60–94% of daily values are 0 (Ujjani 94.2%). Weekly sums inherit this → tiny variance for some reservoirs → NSE/KGE blowups (Ujjani NSE −15.8, Almatti KGE −130).
  - `storage` is also an **input feature** while the target is derived from `storage.diff()` — the model is essentially predicting next-step storage change from current storage. That's near-circular and inflates apparent learnability in training.
  - Storage was reindexed/interpolated to daily from ~2.6-day-spaced source data (2,796–2,819 rows over ~20 years) — a heavy undisclosed transformation.
- **The design doc's core claim (unregulated basin inflows) is currently untested.**

### 🔴 P0-2 — Evaluation is unreliable
1. **Un-scaling bug in `main.py::evaluate`**: targets are *weekly sums of standardized daily values* but are un-scaled with *daily* mean/std (`targets_wk1 * std + mean`). The correct un-scale for a 7-day sum of z-scores is `sum_z * std` (plus 7·mean, and std of a weekly sum ≠ daily std under autocorrelation). This systematically biases every metric and plausibly explains KGE = −130 / −45.9.
2. **Wrong model evaluated**: after early stopping, `train()` returns the *final* weights; `evaluate()` is called *without* a checkpoint, so `runs/best_model_finetune.pt` (best val loss) is never what gets scored.
3. **Config split ignored**: config has `data.val_years: [2023, 2024]`, but `main.py` reads `config["training"]["validation_years"]` — key mismatch → silent fallback to `[2015, 2016, 2023]`. The test year 2024 is then discarded (`train_idx, val_idx, _`). The headline El Niño validation on 2023–24 is not what actually happens.
4. **ENSO stratification on z-scores**: `climate_df` is standardized, so `oni >= 0.5` in `Evaluator` is a z-threshold, not 0.5 °C — the "El Niño vs Neutral" table doesn't mean what it says.
5. **CRPS misuse**: `ps.crps_ensemble` is fed 3 quantiles (P10/P50/P90) as if they were equally-likely ensemble members.
6. **Only week 1 of 12 is evaluated** — the 2–12-week claim is unvalidated.
7. **Stage 2 (mass-balance storage) is never evaluated** — `predict_storage` is dead code; no releases/rule-curve data exists.
8. **No baselines** (climatology, persistence, LSTM, ARIMA) — without persistence you cannot know if the GNN beats "tomorrow = today" on an autocorrelated ΔS target.

### 🔴 P0-3 — The "ENSO×IOD cross-attention" is currently fake
- `combined_climate_indices.csv` columns are `[oni, soi, nino34]` — **no IOD/DMI**.
- `ReservoirGNN.forward` hard-splits `climate_indices[:, :3]` as ENSO and `[:, -1:]` as IOD → **the "IOD" input is Niño3.4 duplicated**. The interaction the design doc centers on (IOD buffering El Niño) cannot be learned because it isn't in the data.
- `data/raw/climate_indices/iod.csv` exists (2005–present) but is never merged into the combined CSV.
- The hard-coded `[:3]` / `[-1:]` split is also fragile: the zero-data fallback builds columns `[ONI, DMI, PDO]`, which changes the meaning of the same slice.

### 🟠 P1-4 — Half the approved architecture is not actually in the pipeline
| Design element | Status |
|---|---|
| Climatological correlation edges (Western-Ghats-aware) | **Never built** — `build_reservoir_graph` is called without `rainfall_data`, so the graph is only ~10 nodes of physical upstream edges; GAT is nearly decorative |
| Static node features (capacity, catchment, elevation, EAC) | **Not used** — model input is 5 dynamic features only; `reservoirs.yaml` metadata sits unused |
| CMIP6/VIC synthetic pre-training | **Absent** — Kaggle "two-phase" just trains twice (epochs=2) on the same observed data; `synthetic_val_loader = val_loader` |
| Lagged ENSO features (1/3/6 months) | Config key `lags_months` ignored; climate tensor is just the 90-day daily window |
| `StratifiedENSOSampler` | Implemented but never used (`shuffle=True` plain loader) |
| Mass-balance stage / rule curves / tribunal releases | Dead code; no release data; no storage evaluation |
| `quantile_weight`, `validation_years`, etc. | Several config keys silently ignored |

### 🟠 P1-5 — Silent-fallback design can train on garbage
- Missing reservoir CSV → filled with `np.random.rand` ("for compilation testing") and training proceeds as normal.
- Missing climate CSV → zero-filled climate indices (a model conditioned on fake constant ENSO).
- Missing ERA5 → zeroed features. Every fallback logs a warning but nothing fails loudly. Results from such runs are indistinguishable from real ones in `runs/`.

### 🟠 P1-6 — Reproducibility & repo hygiene
- **No git repository** — a research project with no version control is one bad `del` from disaster.
- `src/` and `kaggle_stage/src/` have **diverged** (e.g. kaggle `wris_loader` NaNs out *inflow/outflow* zeros too and drops the ΔS derivation) → Kaggle-trained checkpoints and local evaluation use different preprocessing semantics; results are not comparable.
- ~860 MB of zip archives in the root (`kaggle_data.zip` 780 MB, `pls.zip`, `ReservoirGNN_Clean.zip`), `enso_phases_copy.csv` duplicate, broken `kaggle_wris.zip` (0.1 KB), `export/reservoir/json/*.csv` mislabeled files, committed `__pycache__`.
- `requirements.txt` has duplicate entries and no pins/lock; local Python has no torch → the 4 test files can't even run in this workspace; no CI.
- Tests are shape-only smoke tests; no numerical/correctness tests (e.g., pinball loss vs hand-computed values, mass-balance conservation, un-scaling math).

### 🟡 P2-7 — Smaller but real issues
- `wris_loader`: storage `0.0 → NaN → fillna(0)` round-trip; inflow zeros (legit dry-season) vs missing (sensor) are conflated.
- ERA5 `e` evaporation sign ("usually negative, leaving as is") would make mass-balance *add* evaporation.
- `threshold_crossing_accuracy` compares pointwise `obs < t` vs `pred < t` — not the designed "P(storage < 20% by date X)" semantics.
- Per-basin metrics are means of per-reservoir KGEs (mean of −130 and −0.3 is meaningless) — compute on pooled obs/pred per basin instead.
- Climatological-edge "Western Ghats rule" is a single diagonal line — fine as v1, but document it.
- Zero climate rows extend into 2026-12 (future months are placeholders/ffilled).

---

## 4. What can be improved — prioritized roadmap

### P0 — make the results believable (≈ days)
1. **Fix the target.** Either (a) get real inflow/outflow from India-WRIS/NWIC (`download_wris_data.py` already has the NWIC path — pursue it), or (b) honestly reframe the task as *storage/residual forecasting* and update README + design claims. Never leave `inflow = max(ΔS,0)` unlabeled.
2. **Fix evaluation math.** Un-scale weekly sums correctly (`Σz·σ_daily`, no daily-mean offset); evaluate the *best* checkpoint (load `best_model_finetune.pt` before `evaluate`); fix the `val_years` config key; run final numbers on 2024 test only after model selection.
3. **Fix ENSO semantics.** Stratify on raw (un-standardized) ONI; merge `iod.csv` into the combined CSV; index climate columns by name, never by position.
4. **Add baselines now**: persistence (`ΔS_next = ΔS_now`), seasonal climatology, and a per-reservoir LSTM. These are cheap and turn the metrics table into science.

### P1 — make it match the design (≈ 1–2 weeks)
5. Build climatological edges from actual IMD rainfall (pass `rainfall_data`); log edge counts; sanity-check the Ghats rule.
6. Feed static node features (capacity, catchment area, elevation) into the GAT input; they're already in `reservoirs.yaml`.
7. Use `StratifiedENSOSampler`; implement real ENSO-lag features; remove dead config keys or wire them up.
8. Evaluate all 12 weeks (metrics vs horizon plot), and implement the threshold-crossing metric with its designed semantics.
9. Unify `src/` and `kaggle_stage/`: make Kaggle import the same package (e.g. pip-install the repo from the uploaded zip) so there is exactly one source of truth.
10. Replace silent fallbacks with hard failures + an explicit `--allow-synthetic` escape hatch.

### P2 — make it durable (ongoing)
11. `git init` + `.gitignore` (`data/`, `runs/`, zips, `__pycache__`); move archives out of the repo; pin requirements (or `requirements.lock`); add a 3-minute CPU CI running the test suite with tiny synthetic tensors.
12. Add numerical tests: pinball loss vs hand-computed, mass-balance conservation (Σ in = Σ out), un-scaling round-trip, KGE on a known case.
13. Either implement Stage 2 with real release/rule-curve data or demote it in the README to "planned".
14. Verify ERA5 `e` sign convention; document the daily interpolation of WRIS data in the manifest (`created_at` manifest is a great start — add `transformations`).

---

## 5. Bottom line

The design document is publication-grade thinking; the repo is pre-publication scaffolding. Today's `runs/` metrics should not be quoted anywhere: they measure a broken un-scaling path on a ΔS proxy target, scored on the wrong weights, with a graph that has no climatological edges and a "IOD" input that is a duplicated Niño3.4. The good news: every P0 fix is small, well-scoped, and the architecture code itself (GAT/TCN/quantile head) is sound enough to survive the corrections. Fix data + evaluation first; only then will the model's true skill — and the GNN's marginal value over an LSTM baseline — be measurable.
