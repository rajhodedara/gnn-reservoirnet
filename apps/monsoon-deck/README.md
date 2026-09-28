# Monsoon Command Deck

A dark, map-centric data-viz frontend for **ReservoirNet** — next 1–12 week inflow and
storage forecasts for **10 major reservoirs of Peninsular India**.

This is a **static viewer**. There is no backend and no database. The entire app renders
one exported artifact, `public/data/app_data.json`, plus two GeoJSON files. Retrain the
model, re-run the exporter, and the whole deck changes.

```
npm install
npm run dev      # http://localhost:3000
npm run build    # static export lands in out/
```

Requires Node 18+. `npm run build` must pass with no type errors (`npx tsc --noEmit` is clean).

---

## How to swap the JSON when new model runs land

The numbers **will** change after retraining. The app is 100% data-driven — there are zero
hard-coded results in the UI. Follow these three steps.

### 1. Point the exporter at the new run

The exporter reads from fixed repository paths. If the new run lives elsewhere, edit the
path constants at the top of `scripts/export_app_data.py`:

| Constant | Default | Meaning |
|---|---|---|
| `NPZ` | `outputs/rolling_origin/fold_2024/predictions_test.npz` | single-fold forecasts (used for the fan and the frozen tables) |
| `SCOREBOARD_DIR` | `outputs/run9_masked` | **primary** skill scoreboard — per-seed metric CSVs |
| `SCOREBOARD_SEEDS` | `seed42, seed7, seed123, seed2024, seed17` | which seeds get averaged |
| `WRIS_DIR` | `data/raw/wris_v2` | daily inflow/storage per reservoir |
| `BASELINE_JSON` | `outputs/baseline_metrics.json` | persistence + climatology skill |
| `LEVELS_CSV` / `PHYSICS_CSV` | `outputs/level_results`, `outputs/physics_gnn_results` | level/fill skill |
| `ROLLING_CSV` | `outputs/rolling_origin/rolling_origin_summary.csv` | rolling-origin sweep |
| `GEO_DIR` | `data/geo` | India geometry (copied, not regenerated) |

### 2. Re-export

```bash
python scripts/export_app_data.py            # writes apps/monsoon-deck/public/data/app_data.json
python scripts/export_app_data.py --pretty   # indented, for eyeballing a diff
python scripts/export_app_data.py --out /tmp/x.json
```

The script is read-only over every existing artifact: it never mutates the model outputs,
the datasets, or any other script.

It also copies the two GeoJSON files into `public/data/geo/`. Those are real
datameet/geohacker India state boundaries, simplified offline; the exporter does **not**
download or regenerate them.

### 3. Verify, then rebuild

```bash
python scripts/verify_app_data.py    # 1400+ mechanical contract checks; non-zero exit on failure
python scripts/guard_export.py       # fails if the exporter loses a required behaviour
cd apps/monsoon-deck && npm run build
```

`verify_app_data.py` is the important one. It independently re-derives facts rather than
trusting the file:

- greps the raw bytes for `NaN` / `Infinity` (invalid JSON that `json.dump` would happily write)
- recomputes a **7-day daily sum straight from `data/raw/wris_v2/<id>.csv`** and asserts it
  equals `forecast[origin][id].observed[week]` — this is the check that the units and the
  weekly alignment are right, not merely that the JSON parses
- asserts `0 <= p10 <= p50 <= p90` everywhere, and that the clamped-median count the exporter
  discloses matches what the file actually contains
- asserts storage never exceeds gross capacity once TMC/MCM conversion is applied

---

## Data contract

`app_data.json` is fully described by `lib/types.ts`. In brief:

| Key | Contents |
|---|---|
| `meta` | units, split years, band provenance |
| `reservoirs` | the 10 nodes: id, basin, river, state, lat/lon, capacity, dead storage, catchment, tribunal |
| `edges` | 4 directed `physical` reaches + `climatological` teleconnection edges (Pearson rainfall correlation, train period only) |
| `forecast` | `origin date → reservoir → { p10, p50, p90, observed }`, 12 weeks each, 282 origins |
| `forecast_residual_band` | empirical P10/P90 of the model's own held-out residuals, per reservoir per horizon |
| `band_skew` | residual quantiles + clamped-cell disclosure, per reservoir |
| `metrics` | `per_reservoir`, `per_week`, `per_basin`, `enso` (5-seed means with std) + `per_seed` + the frozen `fold_*` tables |
| `integrity` | computed cross-checks and any disclosed discrepancy |
| `baselines` | persistence + climatology NSE/RMSE/MAE per reservoir, plus pooled |
| `levels`, `physics_levels` | level/fill skill at weeks 1/4/12 |
| `rolling_origin` | 2020–2024 sweep, 12 weeks per fold |
| `daily_dates`, `daily`, `era5`, `climate` | observed daily inflow/storage, weather drivers, monthly ONI/SOI/Niño3.4/IOD |
| `cascade` | measured inflow cross-correlation lag per hydraulic reach |

### Units, and why they bite

- **Inflow is a weekly SUM in m³/s·day** — magnitudes run to ~10⁵, so every figure is
  formatted with K/M suffixes.
- **Storage is TMC.** `1 TMC = 28.3168 MCM`. Capacity fill = `storage_tmc / (gross_capacity_mcm / 28.3168)`.
- Weekly alignment, verified to zero error: for origin `o`, week `h` covers the inclusive days
  `[o + 7(h−1), o + 7h − 1]`.
- Genuine zeros are **"no flow", not missing data.** Several reservoirs are dry for 46–67% of
  days and the UI renders those distinctly. Unknown values render as an em dash (`—`) and are
  never filled in.

### Quantile bands are NOT model-issued

This run's export contains only `preds_median`. There is no model P10/P90 anywhere on disk, so
`forecast[*].p10` and `forecast[*].p90` are `null` — deliberately, rather than back-filled with
invented quantiles.

The shaded ribbon you see is `forecast_residual_band`: an empirical P10–P90 of the model's own
held-out residuals, pooled per reservoir and per forecast week across all 282 test origins. It
is a measured error envelope, not a model-issued predictive quantile. The fan chart says so in
its own legend.

If a future export ships real quantiles, the exporter emits them into `p10`/`p90` and the chart
switches to them automatically — no code change needed. The fallback is structural, not a flag.

---

## Honesty is a feature

The model genuinely loses on some dams and in some regimes. The deck does not hide this:

- **The scoreboard shows losses at the same weight as wins.** LOSS chips are rendered next to
  WIN chips; the header counts both (`x/10 beat persistence`, `y/10 beat climatology`).
- **The El Niño row is negative on NSE**, and is annotated as such rather than smoothed away.
- **Level skill is reported as a negative result.** The physics-constrained model beats
  storage persistence on a minority of dams — below the project's own 6/10 bar — and the UI
  states the bar it missed.
- **Where the repository's own files disagree, the disagreement is displayed.** See below.

### Disclosed data discrepancies

The `Provenance & integrity` panel expands to show checks the exporter computes from the real
files. Two currently flag failures, and both are left visible:

1. **`outputs/rolling_origin/fold_2024/` is stale.** Its prediction artifact gives week-1 mean
   NSE of 0.353, while `rolling_origin_summary.csv` claims 0.682 for the same fold. The two
   cannot both be right. Because the **5-seed `outputs/run9_masked/` bundle reproduces the
   repository README headline (mean NSE 0.581, 10/10 dams beating persistence, 7/10 beating
   climatology) to three decimals**, that bundle is used as the primary scoreboard, and the
   single-seed tables are retained under `metrics.fold_*` for reference.
2. **The level summaries do not match the README's operational-mode figure.** The README cites
   0.545 mean week-1 level NSE; `outputs/level_results/summary.csv` gives −6.018 with 1/10 dams
   beating persistence. The panels render the CSVs as found; the README figure evidently comes
   from a different evaluation run not present in this bundle.

Neither number was silently swapped for the friendlier one. If you re-run the pipeline and these
resolve, `verify_app_data.py` will start failing its "discrepancy should currently fail" assertion
— that is the signal to retire the disclosure.

---

## The nine experiences

1. **Network map** — real India geometry, 10 live nodes with capacity-fill rings, 4 animated
   directional river reaches plus the climatological edges.
2. **Horizon scrubber** — week 1→12. Dragging it re-inks the map, the fan, the scoreboard, the
   level table and the cascade together. The confidence ribbon visibly widens.
3. **Quantile fan** — P10–P90 ribbon, P50 line, observed line, JJAS monsoon shading,
   forecast-origin marker.
4. **Skill scoreboard** — sortable; GNN vs persistence vs climatology per dam, win/loss chips,
   basin rollup, El Niño/Neutral pool.
5. **Climate telemetry** — ONI/SOI/Niño3.4/IOD with El Niño and La Niña shading, the forecast
   window marked, and the 2023 onset located from the series.
6. **El Niño / Neutral lens** — re-scopes the deck by the real per-origin ONI classification.
7. **Robustness** — rolling-origin 2020–2024 heatmap, 5 folds × 12 weeks, diverging colour scale.
8. **Level & fill** — storage trajectory vs gross capacity and dead storage, with the honest
   physics result.
9. **Cascade** — upstream pulse vs lag-shifted downstream response, with the measured lag.

## Architecture

```
app/                 Next.js App Router: layout + single route
components/          14 components (map, fan, scoreboard, climate, robustness, levels, cascade, …)
lib/
  types.ts           the data contract
  data.ts            one-shot JSON + GeoJSON load
  store.tsx          React context + useReducer (no Zustand needed)
  selectors.ts       pure derived views — all numbers computed here
  projection.ts      equirectangular projection, GeoJSON→SVG paths, edge geometry
  format.ts          K/M formatting, TMC conversion, monsoon season test
public/data/         app_data.json + geo/
```

Charts are hand-built on `d3-scale` / `d3-shape` (no Recharts) because the quantile ribbon and
the lag-shifted cascade plot need direct path control. The map is plain inline SVG with
`stroke-dashoffset` flow animation.

**Responsive** from mobile up; designed to look right at 1440p. All animations honour
`prefers-reduced-motion`.
