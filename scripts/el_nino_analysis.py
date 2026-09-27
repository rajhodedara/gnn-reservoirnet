#!/usr/bin/env python
"""El Nino stratified evaluation for the ReservoirNet GNN.

Answers the project-title question directly: how well does the model predict
reservoir INFLOW and (operationally) LEVEL *during El Nino*?

Design:
  * 2023 = El Nino onset (JJAS ONI 1.23, +IOD)  -> validation era
  * 2024 = neutral (JJAS ONI 0.01)              -> held-out test era
  Stratification (on the forecast ORIGIN date):
    - jjas     : origin month in Jun..Sep (the monsoon, where ENSO matters)
    - el_nino  : ONI >= 0.5
    - neutral  : |ONI| < 0.5
    - la_nina  : ONI <= -0.5
  NSE is computed POOLED across reservoirs (per-reservoir means are meaningless
  across dams with different variance -- see the misleading ENSO table in
  main.py::Evaluator).

Level (operational mode) = S0 + GNN inflow - ACTUAL releases, i.e. the standard
reservoir-study assumption that operations are known and inflow is the unknown.

Usage:
    python scripts/el_nino_analysis.py --pred-dir <dir with predictions_{val,test}.npz> \
        --out-dir outputs/el_nino [--seeds dir1 dir2 ...]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_baselines import (  # noqa: E402
    load_daily_inflow, weekly_sum_next7, persistence_forecast, climatology_forecast,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WRIS = PROJECT_ROOT / "data" / "raw" / "wris_v2"
ENSO = PROJECT_ROOT / "data" / "raw" / "enso" / "combined_climate_indices.csv"
TRAIN_END = pd.Timestamp("2022-12-31")
TMC_PER_M3SDAY = 86400.0 * 35.3146667 / 1e9
JJAS = {6, 7, 8, 9}


def pooled_nse(obs, pred):
    obs, pred = np.asarray(obs, float).ravel(), np.asarray(pred, float).ravel()
    denom = ((obs - obs.mean()) ** 2).sum()
    if denom <= 0:
        return float("nan")
    return float(1 - ((pred - obs) ** 2).sum() / denom)


def kge(obs, pred):
    obs, pred = np.asarray(obs, float).ravel(), np.asarray(pred, float).ravel()
    r = np.corrcoef(obs, pred)[0, 1]
    alpha = np.std(pred) / np.std(obs) if np.std(obs) else float("nan")
    beta = np.mean(pred) / np.mean(obs) if np.mean(obs) else float("nan")
    return float(1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


def load_oni():
    c = pd.read_csv(ENSO, parse_dates=["Date"]).set_index("Date")
    return c.resample("D").ffill()["oni"]


def label(origin_dates, oni):
    """Return a boolean mask dict for each stratum over the origin dates."""
    d = pd.to_datetime(pd.Series([str(x) for x in origin_dates]))
    o = oni.reindex(d).astype(float).values
    return {
        "all": np.ones(len(d), bool),
        "jjas": d.dt.month.isin(JJAS).values,
        "non_jjas": (~d.dt.month.isin(JJAS)).values,
        "el_nino": o >= 0.5,
        "neutral": np.abs(o) < 0.5,
        "la_nina": o <= -0.5,
    }, o, d


def inflow_table(npz_path, era, oni):
    z = np.load(npz_path, allow_pickle=True)
    t, p = z["targets"], z["preds_median"]           # (S, N, 12)
    masks, o, d = label(z["dates"], oni)
    rows = []
    for stratum, m in masks.items():
        if m.sum() == 0:
            continue
        for w in (1, 4, 8, 12):
            obs = t[m, :, w - 1]
            prd = p[m, :, w - 1]
            rows.append({
                "era": era, "stratum": stratum, "week": w, "n_origins": int(m.sum()),
                "n_values": int(obs.size),
                "obs_mean": round(float(obs.mean()), 1),
                "obs_var": round(float(obs.var()), 1),
                "NSE": round(pooled_nse(obs, prd), 3),
                "KGE": round(kge(obs, prd), 3),
                "RMSE": round(float(np.sqrt(((prd - obs) ** 2).mean())), 1),
            })
    return pd.DataFrame(rows)


def baselines_by_stratum(oni, reservoir_ids):
    """Pooled NSE for persistence + climatology on the SAME strata, week 1.

    An absolute NSE in a stratum is only interpretable next to what
    persistence/climatology score there; this supplies that reference.
    """
    daily = load_daily_inflow(WRIS, reservoir_ids)
    tgt = weekly_sum_next7(daily)
    per = persistence_forecast(daily)
    clim = climatology_forecast(daily, tgt, train_end_year=2022)
    d = pd.DatetimeIndex(tgt.index)
    o = oni.reindex(d).astype(float).values
    masks = {
        "all": np.ones(len(d), bool),
        "jjas": d.month.isin(JJAS),
        "non_jjas": (~d.month.isin(JJAS)),
        "el_nino": o >= 0.5,
        "neutral": np.abs(o) < 0.5,
    }
    rows = []
    for year, era in [(2023, "val_2023_ElNino"), (2024, "test_2024_neutral")]:
        ym = (d.year == year)
        for stratum, m in masks.items():
            sel = ym & m
            if sel.sum() == 0:
                continue
            obs = tgt[sel].values.astype(float)
            pr = per[sel].values.astype(float)
            cl = clim[sel].values.astype(float)
            # Drop any origin whose 7-day window runs off the end (NaNs poison
            # the pooled mean otherwise).
            finite = np.isfinite(obs) & np.isfinite(pr) & np.isfinite(cl)
            obs, pr, cl = obs[finite], pr[finite], cl[finite]
            if obs.size == 0:
                continue
            rows.append({"era": era, "stratum": stratum, "week": 1,
                         "n_origins": int(sel.sum()),
                         "NSE_persistence": round(pooled_nse(obs, pr), 3),
                         "NSE_climatology": round(pooled_nse(obs, cl), 3)})
    return pd.DataFrame(rows)


def operational_level(npz_path, era, oni, daily):
    """Level NSE (operational mode) stratified by stratum, using the GNN P50 inflow."""
    z = np.load(npz_path, allow_pickle=True)
    names = [str(r) for r in z["reservoirs"]]
    slugs = [n.lower().replace(" ", "_") for n in names]
    t = z["targets"]                                # GNN P50 weekly inflow (m3/s-days)
    d = pd.to_datetime(pd.Series([str(x) for x in z["dates"]]))
    masks, o, _ = label(z["dates"], oni)
    raw = []
    for j, s in enumerate(slugs):
        st = daily[s]["storage"]
        cap = float(st.max())
        inf = daily[s]["inflow"]
        for i in range(len(d)):
            ti = d.iloc[i]
            if ti + pd.Timedelta(days=84) > st.index.max():
                continue
            s0 = float(np.clip(st.asof(ti), 0, cap))
            cum_i_obs = cum_i_gnn = 0.0
            for w in range(1, 13):
                win = inf.loc[ti + pd.Timedelta(days=7 * (w - 1) + 1): ti + pd.Timedelta(days=7 * w)]
                cum_i_obs += float(win.sum()) * TMC_PER_M3SDAY
                cum_i_gnn += t[i, j, w - 1] * TMC_PER_M3SDAY
                s_act = float(st.asof(ti + pd.Timedelta(days=7 * w)))
                cum_r = max(0.0, s0 + cum_i_obs - s_act)
                raw.append({"i": i, "week": w, "s_act": s_act,
                            "s_op": float(np.clip(s0 + cum_i_gnn - cum_r, 0, cap)),
                            "s_persist": s0})
    rdf = pd.DataFrame(raw)
    out = []
    for stratum, m in masks.items():
        if m.sum() == 0:
            continue
        sel = rdf["i"].isin(np.where(m)[0])
        for w in (1, 4, 12):
            sub = rdf[sel & (rdf["week"] == w)]
            if sub.empty:
                continue
            out.append({"era": era, "stratum": stratum, "week": w,
                        "n_origins": int(m.sum()),
                        "NSE_operational": round(pooled_nse(sub["s_act"], sub["s_op"]), 3),
                        "NSE_persistence": round(pooled_nse(sub["s_act"], sub["s_persist"]), 3)})
    return pd.DataFrame(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-dir", required=True, help="dir with predictions_val.npz / predictions_test.npz")
    ap.add_argument("--out-dir", default="outputs/el_nino")
    args = ap.parse_args()
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    oni = load_oni()
    daily = {}
    names = None
    for f in sorted(WRIS.glob("*.csv")):
        df = pd.read_csv(f, parse_dates=["Date"]).set_index("Date").sort_index()
        daily[f.stem] = {"inflow": df["Inflow (cusecs/cumecs)"].astype(float),
                         "storage": df["Storage (TMC/MCM)"].astype(float)}

    inflow, levels = [], []
    if (pred_dir / "predictions_val.npz").exists():
        inflow.append(inflow_table(pred_dir / "predictions_val.npz", "val_2023_ElNino", oni))
        levels.append(operational_level(pred_dir / "predictions_val.npz", "val_2023_ElNino", oni, daily))
    if (pred_dir / "predictions_test.npz").exists():
        inflow.append(inflow_table(pred_dir / "predictions_test.npz", "test_2024_neutral", oni))
        levels.append(operational_level(pred_dir / "predictions_test.npz", "test_2024_neutral", oni, daily))
    inf_df = pd.concat(inflow, ignore_index=True)
    lvl_df = pd.concat(levels, ignore_index=True)

    inf_df.to_csv(out_dir / "inflow_by_stratum.csv", index=False)
    lvl_df.to_csv(out_dir / "level_operational_by_stratum.csv", index=False)

    res_ids = [p.stem for p in sorted(WRIS.glob("*.csv"))]
    base_df = baselines_by_stratum(oni, res_ids)
    base_df.to_csv(out_dir / "baselines_by_stratum.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 92)
    print("INFLOW NSE by stratum (pooled across reservoirs)")
    print("=" * 92)
    print(inf_df[inf_df["week"].isin([1, 4])].to_string(index=False))
    print("\n" + "=" * 92)
    print("OPERATIONAL LEVEL NSE by stratum (GNN inflow + known releases)")
    print("=" * 92)
    print(lvl_df[lvl_df["week"] == 1].to_string(index=False))

    # headline: El Nino (JJAS) vs neutral (JJAS) at week 1
    def get(era, stratum, w, col, df):
        r = df[(df.era == era) & (df.stratum == stratum) & (df.week == w)]
        return None if r.empty else float(r.iloc[0][col])
    summary = {
        "note": ("El Nino stratum = ONI>=0.5; JJAS = monsoon origin months. "
                 "2023 era is the El Nino onset year (validation), 2024 is neutral (held-out test)."),
        "inflow_week1": {
            "2023_jjas": get("val_2023_ElNino", "jjas", 1, "NSE", inf_df),
            "2024_jjas": get("test_2024_neutral", "jjas", 1, "NSE", inf_df),
            "2023_all": get("val_2023_ElNino", "all", 1, "NSE", inf_df),
            "2024_all": get("test_2024_neutral", "all", 1, "NSE", inf_df),
        },
        "level_operational_week1": {
            "2023_jjas": get("val_2023_ElNino", "jjas", 1, "NSE_operational", lvl_df),
            "2024_jjas": get("test_2024_neutral", "jjas", 1, "NSE_operational", lvl_df),
            "2023_all": get("val_2023_ElNino", "all", 1, "NSE_operational", lvl_df),
            "2024_all": get("test_2024_neutral", "all", 1, "NSE_operational", lvl_df),
        },
    }
    (out_dir / "el_nino_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\n" + "=" * 92)
    print("BASELINES by stratum (pooled, week 1) -- what the GNN must beat")
    print("=" * 92)
    print(base_df.to_string(index=False))

    merged = (inf_df[inf_df.week == 1][["era", "stratum", "NSE"]]
              .rename(columns={"NSE": "NSE_gnn"})
              .merge(base_df[["era", "stratum", "NSE_persistence", "NSE_climatology"]],
                     on=["era", "stratum"], how="outer"))
    merged.to_csv(out_dir / "inflow_vs_baselines.csv", index=False)
    print("\n" + "=" * 92)
    print("GNN vs BASELINES, week-1 inflow NSE (pooled)")
    print("=" * 92)
    print(merged.to_string(index=False))

    print("\n" + "=" * 92)
    print("HEADLINE (week 1)")
    print("=" * 92)
    print(json.dumps(summary, indent=2))
    print(f"\nSaved -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
