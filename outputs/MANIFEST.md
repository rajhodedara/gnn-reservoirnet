# outputs/ MANIFEST — which file is trustworthy

The `outputs/` tree accumulates artifacts from **several generations of code**.
This file records, for each artifact, which run produced it and whether it may be
quoted. If a file is not listed here as FRESH, treat its numbers as historical.

Provenance keys:
- **FIXED-KAGGLE (2026-09-28)** — produced after the correctness fix-set (`41fda19`
  onwards: config plumbing, climate lags, ENSO gating, graph rebuild, pooled metrics).
- **OLD — do not quote** — produced before those fixes.

---

## FRESH — safe to quote

| Artifact | Source | Notes |
|---|---|---|
| `runs/seed*/` (per-seed dirs) | FIXED-KAGGLE | per-seed metrics + checkpoints, written via `--output-dir` |
| `runs/seed42/evaluation_metrics_pooled_test.csv` | FIXED-KAGGLE | pooled in-domain score (NSE 0.676) |
| `runs/seed42/evaluation_metrics_enso_pooled_test.csv` | FIXED-KAGGLE | pooled El Nino vs Neutral |
| `runs/seed42/predictions_{val,test}.npz` | FIXED-KAGGLE | includes `preds_all` (P10/P50/P90) |
| `ml_baselines.json` | FIXED-KAGGLE | ARIMA/RF/LSTM + re-derived persistence & climatology. **Verified against a fresh local recompute.** |
| `el_nino/` (all files) | FIXED-KAGGLE | phase-stratified inflow + operational-level tables, with baselines on the same strata |
| `level_rulecurve/` | FIXED-KAGGLE | self-contained level model (rule-curve releases) |
| `rolling_origin/fold_{2015,2016,2023,2024}/` | FIXED-KAGGLE | 2015/2016 = design-mandated severe El Nino folds |
| `rolling_origin/rolling_origin_summary.csv` | FIXED-KAGGLE | WARNING: the column `pooled_mean_NSE` is **misnamed** — it is the mean of per-dam NSE. Use `week_*` columns or recompute pooled. |
| `FINAL_phase_table.csv` | FIXED-KAGGLE | the phase table used in the README |

## OLD — do not quote

| Artifact | Why |
|---|---|
| `baseline_metrics.json` | **Stale.** Reports persistence pooled 0.394; a fresh local recompute gives **0.486**. Superseded by `ml_baselines.json`. |
| `seed_summary.csv` | Reports `seeds=3` and a mean that does not match the 5-seed runs (e.g. NS 0.413 vs the correct 0.538). |
| `run1/` ... `run9_masked/` | Archived runs from before the correctness fixes. Historical only. |
| `level_results/`, `physics_gnn_results/`, `figures/` | Mixed-age; regenerate before quoting. |
| `blend_eval_full.csv` | Produced alongside the old scoreboard. |

---

## How to regenerate everything cleanly

```bash
# 1. GNN training + honest evaluation (Kaggle GPU)
#    -> kaggle_runner.ipynb        (5 seeds, per-seed output dirs)
# 2. ML baselines, El Nino folds, level model (Kaggle GPU)
#    -> kaggle_tier3_gpu.ipynb     (LSTM seed-averaged, folds 2015/2016/2023)
# 3. Consolidated tables from the retrieved artifacts
python scripts/definitive_tables.py
```

## Verification performed (2026-09-28)

- fold CSVs agree with an independent recompute from their own `predictions_test.npz` (all four folds, to 3 d.p.)
- `ml_baselines.json` persistence/climatology matches a from-scratch local recompute
- fold 2024's split was checked against the canonical run (identical 282 origins, 2024-01-01 .. 2024-10-08)
- GNN pooled and baselines were re-scored on the **same** origins before comparison
