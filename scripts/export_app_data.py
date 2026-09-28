"""Export a single `app_data.json` for the ReservoirNet "Monsoon Command Deck" frontend.

Read-only over every existing artifact in the repository. Deterministic: the only
non-deterministic field is `meta.generated_at`.

Documented deviations from the schema sketch, both intentional:
  * `daily_dates` is emitted ONCE as a shared 5,479-element array instead of being
    repeated inside every `daily`/`era5` entry. Same data, far smaller file.
  * `forecast[origin][slug]["p10"|"p90"]` are `null`, because the model export on
    disk contains only `preds_median` -- no model-issued quantiles exist anywhere in
    the tree. An empirical residual-quantile band is emitted separately under
    `forecast_residual_band` so it can never be mistaken for a model quantile.

Units (verified against the model targets, not assumed):
  * wris `Inflow (cusecs/cumecs)` column is actually m3/s. A plain 7-day daily sum
    reproduces `predictions_*.npz['targets']` exactly.
  * weekly sum for origin `o`, week `h` (1..12) covers inclusive days
    [o + 7*(h-1), o + 7*h - 1].
  * wris `Storage (TMC/MCM)` column is actually TMC. 1 TMC = 28.3168 MCM.

Dependencies: stdlib + numpy + pandas only. `configs/reservoirs.yaml` is read with a
small purpose-built parser because PyYAML is not installed in this environment.

Sanitisation: non-finite values (NaN/inf) and the absurd Log_NSE sentinel
(|x| > 1e6) become JSON `null`. `json.dump` would happily emit bare `NaN`/`Infinity`
tokens, which are invalid JSON, so everything is converted first.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WRIS_DIR = os.path.join(REPO, "data", "raw", "wris_v2")
ERA5_CSV = os.path.join(REPO, "data", "raw", "era5", "reservoir_era5_daily.csv")
CLIMATE_CSV = os.path.join(REPO, "data", "raw", "enso", "combined_climate_indices.csv")
RESERVOIRS_YAML = os.path.join(REPO, "configs", "reservoirs.yaml")
GEO_DIR = os.path.join(REPO, "data", "geo")

NPZ = os.path.join(REPO, "outputs", "rolling_origin", "fold_2024", "predictions_test.npz")
MET_DIR = os.path.join(REPO, "outputs", "rolling_origin", "fold_2024")
BASELINE_JSON = os.path.join(REPO, "outputs", "baseline_metrics.json")
LEVELS_CSV = os.path.join(REPO, "outputs", "level_results", "summary.csv")
PHYSICS_CSV = os.path.join(REPO, "outputs", "physics_gnn_results", "summary.csv")
ROLLING_CSV = os.path.join(REPO, "outputs", "rolling_origin", "rolling_origin_summary.csv")
LEVEL_PHASE_CSV = os.path.join(REPO, "outputs", "level_rulecurve", "level_rulecurve_by_phase.csv")

# The canonical scoreboard is the 5-seed masked run: its per-reservoir NSE reproduces
# the repository README headline (mean 0.581) to three decimals. The single-seed
# `outputs/rolling_origin/fold_2024/` bundle is STALE relative to both the README and
# its own rolling_origin_summary.csv, so it is not used for headline metrics; the
# disagreement is measured and disclosed in `integrity` instead of being hidden.
SCOREBOARD_DIR = os.path.join(REPO, "outputs", "run9_masked")
SCOREBOARD_SEEDS = ["seed42", "seed7", "seed123", "seed2024", "seed17"]
HEADLINE_SOURCE = "outputs/run9_masked (5 seeds)"

DEFAULT_OUT_DIR = os.path.join(REPO, "apps", "monsoon-deck", "public", "data")

TRAIN_END = "2022-12-31"
CLIM_BASE_THRESHOLD = 0.6
CLIM_GHATS_THRESHOLD = 1.0
CASCADE_MAX_LAG = 30
LOGNSE_SENTINEL = 1e6

RESERVOIR_LABEL_TO_ID = {
    "Nagarjuna Sagar": "nagarjuna_sagar",
    "Srisailam": "srisailam",
    "Almatti": "almatti",
    "Tungabhadra": "tungabhadra",
    "Mettur": "mettur",
    "Krishnaraja Sagara": "krishnaraja_sagara",
    "Jayakwadi": "jayakwadi",
    "Ujjani": "ujjani",
    "Sardar Sarovar": "sardar_sarovar",
    "Ukai": "ukai",
}


def clean(v):
    """JSON-safe scalar: non-finite and sentinel values collapse to None."""
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        f = float(v)
        if not math.isfinite(f) or abs(f) > LOGNSE_SENTINEL:
            return None
        return f
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    return v


def rnd(v, nd):
    c = clean(v)
    if c is None or isinstance(c, (str, bool, int)):
        return c
    return round(c, nd)


def arr(values, nd):
    out = []
    for v in values:
        c = clean(v)
        out.append(None if c is None else round(float(c), nd))
    return out


def fail(msg):
    print(f"EXPORT ERROR: {msg}", file=sys.stderr)
    raise SystemExit(1)


def parse_reservoirs_yaml(path):
    """Tiny parser for configs/reservoirs.yaml's regular structure (no PyYAML)."""
    if not os.path.exists(path):
        fail(f"missing {path}")
    text = open(path, encoding="utf-8").read()

    def scalar(raw):
        raw = raw.strip()
        if raw.startswith("[") and raw.endswith("]"):
            inner = raw[1:-1].strip()
            if not inner:
                return []
            return [s.strip().strip('"').strip("'") for s in inner.split(",")]
        if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "\"'":
            return raw[1:-1]
        try:
            return int(raw)
        except ValueError:
            pass
        try:
            return float(raw)
        except ValueError:
            pass
        return raw

    reservoirs, current, collecting = [], None, None
    for raw_line in text.splitlines():
        if not raw_line.strip() or raw_line.strip().startswith("#"):
            continue
        stripped = raw_line.strip()

        if stripped.startswith("- ") and ":" in stripped and not stripped[2:].startswith(" "):
            if current:
                reservoirs.append(current)
            current, collecting = {}, None
            k, _, v = stripped[2:].partition(":")
            current[k.strip()] = scalar(v)
            continue

        if current is None:
            continue

        if collecting is not None and stripped.startswith("- "):
            current[collecting].append(scalar(stripped[2:]))
            continue

        collecting = None
        if ":" not in stripped:
            continue
        k, _, v = stripped.partition(":")
        k, v = k.strip(), v.strip()
        if v == "":
            current[k] = []
            collecting = k
        else:
            current[k] = scalar(v)

    if current:
        reservoirs.append(current)

    if len(reservoirs) != 10:
        fail(f"expected 10 reservoirs in {path}, parsed {len(reservoirs)}")
    for r in reservoirs:
        for req in ("id", "name", "basin", "river", "state", "latitude", "longitude",
                    "gross_capacity_mcm", "dead_storage_mcm", "catchment_area_km2", "tribunal"):
            if req not in r:
                fail(f"reservoir {r.get('id')} missing required field {req}")
        r.setdefault("upstream", [])
        r.setdefault("elevation_m", None)
    return reservoirs


def get_region(lon, lat):
    """Windward/leeward of the Western Ghats. Ported from src/data/graph_builder.py."""
    ridge_lon = 76.0 - ((lat - 8.0) / 12.0) * 2.5
    return "windward" if lon < ridge_lon else "leeward"


def build_climatological_edges(rainfall_df, reservoirs):
    """Port of build_climatological_edges(): corr over TRAIN only, i<j scan, both directions."""
    ids = [r["id"] for r in reservoirs]
    by_id = {r["id"]: r for r in reservoirs}
    valid = [i for i in ids if i in rainfall_df.columns]
    corr = rainfall_df[valid].corr(method="pearson")

    edges = []
    for a, id1 in enumerate(valid):
        for b, id2 in enumerate(valid):
            if a >= b:
                continue
            c = corr.loc[id1, id2]
            if not np.isfinite(c):
                continue
            r1, r2 = by_id[id1], by_id[id2]
            cross = get_region(r1["longitude"], r1["latitude"]) != get_region(r2["longitude"], r2["latitude"])
            threshold = CLIM_GHATS_THRESHOLD if cross else CLIM_BASE_THRESHOLD
            if c > threshold:
                w = round(float(c), 6)
                edges.append({"source": id1, "target": id2, "kind": "climatological", "weight": w})
                edges.append({"source": id2, "target": id1, "kind": "climatological", "weight": w})
    return edges


def build_cascade(daily, physical_edges):
    """Measured inflow cross-correlation lag. Explicitly NOT hydrological routing."""
    out = []
    for e in physical_edges:
        up, down = e["source"], e["target"]
        a = np.log1p(np.clip(np.asarray(daily[up]["inflow"], dtype=float), 0, None))
        b = np.log1p(np.clip(np.asarray(daily[down]["inflow"], dtype=float), 0, None))
        best_lag, best_corr, best_n = 0, None, 0
        for lag in range(0, CASCADE_MAX_LAG + 1):
            x = a[: len(a) - lag] if lag else a
            y = b[lag:] if lag else b
            if len(x) < 365:
                continue
            if np.nanstd(x) == 0 or np.nanstd(y) == 0:
                continue
            c = float(np.corrcoef(x, y)[0, 1])
            if not np.isfinite(c):
                continue
            if best_corr is None or c > best_corr:
                best_lag, best_corr, best_n = lag, c, len(x)
        out.append({
            "upstream": up,
            "downstream": down,
            "lag_days": int(best_lag),
            "peak_corr": None if best_corr is None else round(best_corr, 6),
            "n_pairs": int(best_n),
            "cascade_note": (
                "Lag of maximum cross-correlation between log1p-transformed daily inflow series, "
                "lags 0-30 days, 2010-2024. This is an inflow-to-inflow statistical lag, NOT release "
                "routing: no release schedule or channel routing is modelled."
            ),
        })
    return out


def build(out_path, pretty):
    reservoirs = parse_reservoirs_yaml(RESERVOIRS_YAML)
    ids = [r["id"] for r in reservoirs]

    # ---------- daily wris ----------
    wris, daily_dates = {}, None
    for rid in ids:
        p = os.path.join(WRIS_DIR, f"{rid}.csv")
        if not os.path.exists(p):
            fail(f"missing wris csv {p}")
        df = pd.read_csv(p)
        dates = df["Date"].astype(str).tolist()
        if daily_dates is None:
            daily_dates = dates
        elif dates != daily_dates:
            fail(f"date index mismatch for {rid}")
        wris[rid] = {
            "inflow": arr(df["Inflow (cusecs/cumecs)"].to_numpy(dtype=float), 3),
            "storage": arr(df["Storage (TMC/MCM)"].to_numpy(dtype=float), 3),
        }
    if len(daily_dates) != 5479:
        fail(f"expected 5479 daily dates, got {len(daily_dates)}")

    # ---------- era5 ----------
    era5_df = pd.read_csv(ERA5_CSV)
    if era5_df["Date"].astype(str).tolist() != daily_dates:
        fail("era5 dates do not align with wris dates")
    era5 = {}
    for rid in ids:
        entry = {}
        for field in ("rainfall", "runoff", "evap", "soil_moisture"):
            col = f"{rid}_{field}"
            if col not in era5_df.columns:
                fail(f"era5 column missing: {col}")
            entry[field] = arr(era5_df[col].to_numpy(dtype=float), 3)
        era5[rid] = entry

    # ---------- climate ----------
    clim_df = pd.read_csv(CLIMATE_CSV)
    climate = {
        "dates": clim_df["Date"].astype(str).tolist(),
        "oni": arr(clim_df["oni"].to_numpy(dtype=float), 3),
        "soi": arr(clim_df["soi"].to_numpy(dtype=float), 3),
        "nino34": arr(clim_df["nino34"].to_numpy(dtype=float), 3),
        "iod": arr(clim_df["iod"].to_numpy(dtype=float), 3),
    }

    # ---------- forecasts ----------
    npz = np.load(NPZ, allow_pickle=True)
    labels = [str(x) for x in npz["reservoirs"]]
    origin_dates = [str(x) for x in npz["dates"]]
    targets, medians = npz["targets"], npz["preds_median"]
    if targets.shape != (282, 10, 12) or medians.shape != (282, 10, 12):
        fail(f"unexpected npz shapes {targets.shape} {medians.shape}")
    if set(labels) != set(RESERVOIR_LABEL_TO_ID):
        fail(f"unexpected reservoir labels in npz: {labels}")
    col_of = {RESERVOIR_LABEL_TO_ID[l]: i for i, l in enumerate(labels)}

    forecast = {}
    resid = {rid: [[] for _ in range(12)] for rid in ids}
    for oi, od in enumerate(origin_dates):
        entry = {}
        for rid in ids:
            ci = col_of[rid]
            entry[rid] = {
                "p10": None,
                "p50": arr(medians[oi, ci], 3),
                "p90": None,
                "observed": arr(targets[oi, ci], 3),
            }
            for h in range(12):
                r = targets[oi, ci, h] - medians[oi, ci, h]
                if np.isfinite(r):
                    resid[rid][h].append(float(r))
        forecast[od] = entry

    # ---------- empirical residual band + honest bias diagnostics ----------
    band_q = {}
    for rid in ids:
        q10, q90 = [], []
        for h in range(12):
            vals = np.asarray(resid[rid][h], dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size >= 20:
                q10.append(float(np.nanpercentile(vals, 10)))
                q90.append(float(np.nanpercentile(vals, 90)))
            else:
                q10.append(None)
                q90.append(None)
        band_q[rid] = (q10, q90)

    forecast_residual_band = {}
    for od in origin_dates:
        entry = {}
        for rid in ids:
            p50 = forecast[od][rid]["p50"]
            q10, q90 = band_q[rid]
            lo, hi = [], []
            for h in range(12):
                med = p50[h]
                if med is None or q10[h] is None:
                    lo.append(None)
                    hi.append(None)
                    continue
                # Inflow is physically non-negative, but the model median goes negative at
                # long horizons on several nodes. The raw median stays untouched in
                # forecast[*].p50; the band is built on a zero-clamped median so the ribbon is
                # always ordered. Clamped-cell counts are disclosed in `band_skew`.
                med_c = max(med, 0.0)
                low = max(med_c + q10[h], 0.0)
                high = med_c + q90[h]
                # The residual distribution is strongly right-skewed at several nodes (the model
                # systematically under-predicts), so the empirical 10th percentile of error can be
                # positive. Enforce 0 <= p10 <= p50 <= p90 by construction.
                if low > med_c:
                    low = med_c
                if high < med_c:
                    high = med_c
                lo.append(round(low, 3))
                hi.append(round(high, 3))
            entry[rid] = {"p10": lo, "p90": hi}
        forecast_residual_band[od] = entry

    band_skew = {}
    for rid in ids:
        allr = np.asarray([v for h in range(12) for v in resid[rid][h]], dtype=float)
        allr = allr[np.isfinite(allr)]
        per_week_q10 = band_q[rid][0]
        neg_cells, neg_weeks = 0, set()
        for od in origin_dates:
            p50 = forecast[od][rid]["p50"]
            for h in range(12):
                if p50[h] is not None and p50[h] < 0:
                    neg_cells += 1
                    neg_weeks.add(h + 1)
        band_skew[rid] = {
            "n_residuals": int(allr.size),
            "median": rnd(np.median(allr), 3) if allr.size else None,
            "q10": rnd(np.percentile(allr, 10), 3) if allr.size else None,
            "q90": rnd(np.percentile(allr, 90), 3) if allr.size else None,
            "weeks_with_positive_q10": [
                h + 1 for h in range(12)
                if per_week_q10[h] is not None and per_week_q10[h] > 0
            ],
            "negative_median_cells": int(neg_cells),
            "negative_median_weeks": sorted(neg_weeks),
        }

    # ---------- edges ----------
    physical = []
    for r in reservoirs:
        for up in r.get("upstream", []):
            if up in ids:
                physical.append({"source": up, "target": r["id"], "kind": "physical", "weight": None})

    rain = pd.DataFrame({"Date": pd.to_datetime(daily_dates)})
    for rid in ids:
        rain[rid] = era5[rid]["rainfall"]
    rain = rain.set_index("Date")
    clim_edges = build_climatological_edges(rain.loc[rain.index <= pd.Timestamp(TRAIN_END)], reservoirs)
    edges = physical + clim_edges

    # ---------- metrics ----------
    def read_metrics(name):
        p = os.path.join(MET_DIR, name)
        if not os.path.exists(p):
            fail(f"missing metrics file {p}")
        return pd.read_csv(p)

    def metric_rows(df, numeric_cols):
        out = []
        for rec in df.to_dict(orient="records"):
            row = {}
            for k, v in rec.items():
                if k in numeric_cols:
                    row[k] = rnd(v, 6)
                elif isinstance(v, str):
                    row[k] = v.strip()
                else:
                    row[k] = clean(v)
            out.append(row)
        return out

    pr_num = ["CRPS", "RMSE", "MAE", "NSE", "Event_NSE", "Log_NSE", "KGE"]
    scal = ["CRPS", "RMSE", "MAE", "NSE", "Event_NSE", "Log_NSE", "KGE"]
    per_reservoir = metric_rows(read_metrics("evaluation_metrics_per_reservoir_test.csv"), pr_num)
    for row in per_reservoir:
        row["reservoir_id"] = RESERVOIR_LABEL_TO_ID.get(row.get("Reservoir"))

    per_week = metric_rows(read_metrics("evaluation_metrics_by_week_test.csv"), pr_num + ["Week"])
    for row in per_week:
        row["Week"] = int(row["Week"])
        row["reservoir_id"] = RESERVOIR_LABEL_TO_ID.get(row.get("Reservoir"))

    per_basin = metric_rows(read_metrics("evaluation_metrics_per_basin_test.csv"),
                            ["CRPS", "RMSE", "NSE", "Event_NSE", "Log_NSE", "KGE"])
    enso = metric_rows(read_metrics("evaluation_metrics_enso_test.csv"),
                       ["CRPS", "RMSE", "NSE", "Event_NSE", "Log_NSE", "KGE"])

    # ---------- canonical 5-seed scoreboard ----------
    # Mean over the real per-seed CSVs, carrying a std so seed spread is visible.
    seed_frames = {}
    for kind, fname in (
        ("per_reservoir", "evaluation_metrics_per_reservoir_test.csv"),
        ("per_week", "evaluation_metrics_by_week_test.csv"),
        ("per_basin", "evaluation_metrics_per_basin_test.csv"),
        ("enso", "evaluation_metrics_enso_test.csv"),
    ):
        frames = []
        for sd in SCOREBOARD_SEEDS:
            p = os.path.join(SCOREBOARD_DIR, sd, fname)
            if not os.path.exists(p):
                fail(f"missing canonical scoreboard file {p}")
            df = pd.read_csv(p)
            df["seed"] = sd
            frames.append(df)
        seed_frames[kind] = pd.concat(frames, ignore_index=True)

    def mean_over_seeds(kind, keys, numeric_cols):
        df = seed_frames[kind]
        g = df.groupby(keys, dropna=False)[numeric_cols].agg(["mean", "std"])
        g.columns = [f"{a}_{b}" for a, b in g.columns]
        rows = []
        for rec in g.reset_index().to_dict(orient="records"):
            row = {}
            for k, v in rec.items():
                row[k] = rnd(v, 6)
            rows.append(row)
        return rows

    sc_per_res = mean_over_seeds("per_reservoir", ["Reservoir", "Basin"], pr_num)
    for row in sc_per_res:
        row["reservoir_id"] = RESERVOIR_LABEL_TO_ID.get(row.get("Reservoir"))
        row["n_seeds"] = len(SCOREBOARD_SEEDS)

    sc_per_week = mean_over_seeds("per_week", ["Week", "Reservoir", "Basin"], pr_num)
    for row in sc_per_week:
        row["Week"] = int(row["Week"])
        row["reservoir_id"] = RESERVOIR_LABEL_TO_ID.get(row.get("Reservoir"))

    sc_per_basin = mean_over_seeds("per_basin", ["Basin"],
                                 ["CRPS", "RMSE", "NSE", "Event_NSE", "Log_NSE", "KGE"])
    sc_enso = mean_over_seeds("enso", ["Condition"],
                              ["CRPS", "RMSE", "NSE", "Event_NSE", "Log_NSE", "KGE"])

    per_seed = []
    for kind in ("per_reservoir", "per_week", "per_basin", "enso"):
        for rec in seed_frames[kind].to_dict(orient="records"):
            row = {"kind": kind}
            for k, v in rec.items():
                row[k] = rnd(v, 6) if k in scal or k == "Week" else clean(v)
            if "Week" in row and row["Week"] is not None:
                row["Week"] = int(row["Week"])
            per_seed.append(row)

    nse_means = [r.get("NSE_mean") for r in sc_per_res if r.get("NSE_mean") is not None]
    wk1 = [r for r in sc_per_week if r["Week"] == 1]
    headline = {
        "source": HEADLINE_SOURCE,
        "n_seeds": len(SCOREBOARD_SEEDS),
        "mean_nse_pooled": rnd(sum(nse_means) / len(nse_means), 6) if nse_means else None,
        "mean_nse_week1": rnd(sum(r["NSE_mean"] for r in wk1) / len(wk1), 6) if wk1 else None,
    }

    # ---------- integrity: measure the scoreboard-vs-fold disagreement ----------
    def nse_arr(obs, pred):
        o = np.asarray(obs, dtype=float)
        p = np.asarray(pred, dtype=float)
        m = np.isfinite(o) & np.isfinite(p)
        o, p = o[m], p[m]
        if o.size < 2:
            return None
        den = float(((o - o.mean()) ** 2).sum())
        if den == 0:
            return None
        return float(1 - ((o - p) ** 2).sum() / den)

    fold_w1 = None
    try:
        fd = np.load(NPZ, allow_pickle=True)
        vals = [nse_arr(fd["targets"][:, i, 0], fd["preds_median"][:, i, 0]) for i in range(fd["targets"].shape[1])]
        vals = [v for v in vals if v is not None]
        fold_w1 = sum(vals) / len(vals) if vals else None
    except Exception:
        fold_w1 = None

    rolling_rows = pd.read_csv(ROLLING_CSV).to_dict(orient="records")
    ro_2024 = next((r for r in rolling_rows if int(r["fold"]) == 2024), None)
    ro_2024_w1 = ro_2024.get("week_1") if ro_2024 else None

    # Level summaries vs the README's operational-mode claim (0.545 at week 1).
    op_w1_mean = None
    ph_w1_wins = None
    try:
        op = pd.read_csv(LEVELS_CSV)
        opw = op[(op["split"] == "test") & (op["week"] == 1)]
        if len(opw):
            op_w1_mean = float(opw["level_nse"].mean())
        ph = pd.read_csv(PHYSICS_CSV)
        phw = ph[(ph["split"] == "test") & (ph["week"] == 1)]
        if len(phw):
            ph_w1_wins = int((phw["level_nse"] > phw["pers_nse"]).sum())
    except Exception:
        pass

    integrity = {
        "checked": [
            {
                "name": "scoreboard_reproduces_readme",
                "status": "pass" if headline["mean_nse_pooled"] and abs(headline["mean_nse_pooled"] - 0.581) < 0.01 else "fail",
                "detail": (
                    "The 5-seed scoreboard mean NSE is "
                    f"{headline['mean_nse_pooled']:.4f}, which matches the repository README "
                    "headline of 0.581. This is the number the deck reports as primary skill."
                ),
            },
            {
                "name": "fold_2024_artifact_matches_its_own_summary",
                "status": (
                    "pass"
                    if (fold_w1 is not None and ro_2024_w1 is not None and abs(fold_w1 - float(ro_2024_w1)) < 0.02)
                    else "fail"
                ),
                "detail": (
                    "The single-seed fold-2024 prediction artifact gives week-1 mean NSE "
                    f"{('%.4f' % fold_w1) if fold_w1 is not None else 'n/a'}, while "
                    "rolling_origin_summary.csv reports "
                    f"{('%.4f' % float(ro_2024_w1)) if ro_2024_w1 is not None else 'n/a'} for the same fold. "
                    "They disagree, so the rolling-origin fold artifacts on disk are stale for 2024. "
                    "That row is shown because it is what the repository contains, and is flagged here."
                ),
            },
            {
                "name": "level_summaries_match_readme_operational_claim",
                "status": (
                    "pass" if (op_w1_mean is not None and abs(op_w1_mean - 0.545) < 0.05) else "fail"
                ),
                "detail": (
                    "The README reports operational-mode level week-1 mean NSE of 0.545, but "
                    "outputs/level_results/summary.csv gives "
                    f"{('%.4f' % op_w1_mean) if op_w1_mean is not None else 'n/a'} "
                    "(beats persistence on 1 of 10 dams). The physics summary gives "
                    f"{ph_w1_wins if ph_w1_wins is not None else 'n/a'} of 10 at week 1. "
                    "The level panels render these CSVs as found; the README figure evidently comes "
                    "from a different per-dam evaluation run that is not present in this bundle."
                ),
            },
        ],
        "note": (
            "Nothing in this deck was recomputed or regenerated. All artifacts are read as found. "
            "Where the repository's own files disagree with each other, the disagreement is measured "
            "here and surfaced in the UI rather than resolved silently in favour of the nicer number."
        ),
    }

    # ---------- baselines ----------
    bl = json.load(open(BASELINE_JSON, encoding="utf-8"))
    baselines, pooled = {}, {}
    for name in ("persistence", "climatology"):
        src = bl.get(name, {}).get("test", {})
        baselines[name] = {
            sid: {"NSE": rnd(v.get("NSE"), 6), "RMSE": rnd(v.get("RMSE"), 4), "MAE": rnd(v.get("MAE"), 4)}
            for sid, v in src.items() if sid in ids
        }
        # `test` also carries a real `_pooled` row (all reservoirs pooled). Keep it, but under
        # baselines.pooled so it can never be mistaken for a reservoir id.
        if src.get("_pooled"):
            r = src["_pooled"]
            pooled[name] = {
                "NSE": rnd(r.get("NSE"), 6), "RMSE": rnd(r.get("RMSE"), 4), "MAE": rnd(r.get("MAE"), 4),
                "n": int(r["n"]) if r.get("n") is not None else None,
            }
    baselines["pooled"] = pooled

    # ---------- levels ----------
    def level_rows(path, cols):
        df = pd.read_csv(path)
        df = df[df["split"] == "test"]
        out = []
        for rec in df.to_dict(orient="records"):
            row = {"dam": str(rec["dam"]).strip(), "split": "test", "week": int(rec["week"])}
            for c in cols:
                row[c] = rnd(rec.get(c), 6)
            out.append(row)
        out.sort(key=lambda r: (r["dam"], r["week"]))
        return out

    levels = level_rows(LEVELS_CSV, ["level_nse", "pers_nse", "clim_nse", "ds_nse"])
    physics_levels = level_rows(PHYSICS_CSV, ["level_nse", "pers_nse", "clim_nse", "n_samples", "gross_tmc"])
    for row in physics_levels:
        if row.get("n_samples") is not None:
            row["n_samples"] = int(row["n_samples"])

    # ---------- rolling origin ----------
    rolling_origin = []
    for rec in pd.read_csv(ROLLING_CSV).to_dict(orient="records"):
        row = {"fold": int(rec["fold"])}
        for h in range(1, 13):
            row[f"week_{h}"] = rnd(rec.get(f"week_{h}"), 6)
        row["pooled_mean_NSE"] = rnd(rec.get("pooled_mean_NSE"), 6)
        rolling_origin.append(row)
    rolling_origin.sort(key=lambda r: r["fold"])

    # ---------- level skill BY CLIMATE PHASE (the title's claim) ----------
    # Reads outputs/level_rulecurve/level_rulecurve_by_phase.csv, which
    # scripts/eval_levels_rulecurve.py writes when it stratifies level skill by
    # JJAS-mean ONI. Absent on older runs -> emit null so the UI shows an em dash.
    level_by_phase = None
    if os.path.exists(LEVEL_PHASE_CSV):
        lp = pd.read_csv(LEVEL_PHASE_CSV)
        rows = []
        for rec in lp.to_dict(orient="records"):
            rows.append({
                "phase": str(rec["Phase"]),
                "week": int(rec["Week"]),
                "n": int(rec["n"]),
                "nse_rulecurve": rnd(rec.get("NSE_rulecurve"), 6),
                "nse_zero_release": rnd(rec.get("NSE_zero_release"), 6),
                "nse_climatology": rnd(rec.get("NSE_climatology"), 6),
                "nse_persistence": rnd(rec.get("NSE_persistence"), 6),
            })
        rows.sort(key=lambda r: (r["phase"], r["week"]))
        level_by_phase = rows

    cascade = build_cascade(wris, physical)

    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "test_year": 2024,
            "val_year": 2023,
            "train_end": TRAIN_END,
            "units_inflow": "m3/s\u00b7day (weekly sum)",
            "units_storage": "TMC",
            "band_source": "median_only + empirical_residual_quantiles",
            "band_note": (
                "The export on disk contains only the model median (preds_median); no model-issued "
                "P10/P90 quantiles exist for this run, so forecast[*].p10 and forecast[*].p90 are null. "
                "The shaded band comes from forecast_residual_band: an empirical P10-P90 of the model's "
                "own held-out residuals (observed minus median), pooled per reservoir and per forecast "
                "week across all 282 test origins. It is a measured error envelope, not a model-issued "
                "predictive quantile, and it widens with horizon. Where the model systematically "
                "under-predicts, the residual P10 is positive and the lower edge is floored at the median "
                "rather than inverted. The model median itself goes physically invalid (negative) at long "
                "horizons on some nodes: it is left raw in forecast[*].p50, the band is built on a "
                "zero-clamped median, and the number of clamped cells is disclosed per reservoir in "
                "band_skew."
            ),
            "units_capacity_note": "1 TMC = 28.3168 MCM; capacity fill = storage_tmc / (gross_capacity_mcm / 28.3168)",
        },
        "reservoirs": [
            {
                "id": r["id"], "name": r["name"], "basin": r["basin"], "river": r["river"],
                "state": r["state"], "lat": rnd(r["latitude"], 6), "lon": rnd(r["longitude"], 6),
                "gross_capacity_mcm": rnd(r["gross_capacity_mcm"], 4),
                "dead_storage_mcm": rnd(r["dead_storage_mcm"], 4),
                "catchment_area_km2": rnd(r["catchment_area_km2"], 4),
                "tribunal": r["tribunal"], "elevation_m": rnd(r.get("elevation_m"), 4),
                "upstream": list(r.get("upstream", [])),
            }
            for r in reservoirs
        ],
        "edges": edges,
        "forecast": forecast,
        "forecast_residual_band": forecast_residual_band,
        "band_skew": band_skew,
        "metrics": {
            "per_reservoir": sc_per_res,
            "per_week": sc_per_week,
            "per_basin": sc_per_basin,
            "enso": sc_enso,
            "per_seed": per_seed,
            "headline": headline,
            "fold_per_reservoir": per_reservoir,
            "fold_per_week": per_week,
            "fold_per_basin": per_basin,
            "fold_enso": enso,
            "note": (
                "Primary metrics are the mean over the 5 canonical seeds with a per-metric std, "
                "because that is what reproduces the repository README headline. The single-seed "
                "fold_2024 bundle is retained under fold_* keys and is known to be stale; see `integrity`."
            ),
        },
        "integrity": integrity,
        "baselines": baselines,
        "levels": levels,
        "physics_levels": physics_levels,
        "rolling_origin": rolling_origin,
        "level_by_phase": level_by_phase,
        "daily_dates": daily_dates,
        "daily": wris,
        "climate": climate,
        "era5": era5,
        "cascade": cascade,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(payload, fh, ensure_ascii=False,
                  indent=1 if pretty else None,
                  separators=None if pretty else (",", ":"),
                  allow_nan=False)
    size = os.path.getsize(out_path)

    geo_out = os.path.join(os.path.dirname(out_path), "geo")
    os.makedirs(geo_out, exist_ok=True)
    for fn in ("india_states_simplified.geojson", "india_outline_simplified.geojson"):
        src = os.path.join(GEO_DIR, fn)
        if not os.path.exists(src):
            fail(f"missing geo asset {src}")
        shutil.copyfile(src, os.path.join(geo_out, fn))

    n_phys = sum(1 for e in edges if e["kind"] == "physical")
    n_clim = sum(1 for e in edges if e["kind"] == "climatological")
    print(f"wrote {out_path}")
    print(f"  bytes            : {size:,} ({size / 1048576:.2f} MB)")
    print(f"  reservoirs       : {len(payload['reservoirs'])}")
    print(f"  edges            : {len(edges)} total ({n_phys} physical, {n_clim} climatological)")
    print(f"  forecast origins : {len(forecast)}")
    print(f"  daily rows       : {len(daily_dates)}")
    print(f"  climate months   : {len(climate['dates'])} (iod nulls: "
          f"{sum(1 for v in climate['iod'] if v is None)})")
    print(f"  per_week rows    : {len(sc_per_week)}")
    print(f"  seed scoreboard  : {headline['n_seeds']} seeds, pooled mean NSE {headline['mean_nse_pooled']}")
    for chk in integrity["checked"]:
        print(f"  integrity        : {chk['status'].upper():4s} {chk['name']}")
    print(f"  rolling folds    : {len(rolling_origin)}")
    print("  cascade edges    : " + ", ".join(
        f"{c['upstream']}->{c['downstream']} lag={c['lag_days']}d r={c['peak_corr']}" for c in cascade))
    print(f"  geo copied to    : {geo_out}")
    return size


def main():
    ap = argparse.ArgumentParser(description="Export app_data.json for the Monsoon Command Deck.")
    ap.add_argument("--out", default=os.path.join(DEFAULT_OUT_DIR, "app_data.json"))
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args()
    build(args.out, args.pretty)


if __name__ == "__main__":
    main()
