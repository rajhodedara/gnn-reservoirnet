#!/usr/bin/env python
"""Self-contained LEVEL forecast via an empirical rule-curve release model.

The approved design specifies Stage 2 as a deterministic mass balance

    S(t+1) = S(t) + I(t) - E(t) - R(t)

with R from CWC rule curves / tribunal mandates. The ML release head that was
tried instead does not work (strongly negative self-contained level NSE), because
releases are governed by policy, not by learnable inflow physics.

This script implements the design's *rule curve* honestly:

  1. Recover implied historical releases from mass balance on the TRAINING
     window only:   R_implied(t) = S(t) + I(t) - S(t+1)  (clipped >= 0).
  2. Fit an empirical rule curve  R_hat = f(storage_fraction, day_of_year)
     on those training releases (gradient boosting, per dam).
  3. Roll the mass balance forward with the GNN's predicted inflow and the
     rule-curve release (no future knowledge) to get a self-contained level
     forecast.
  4. Score storage NSE vs persistence, climatology, and the naive zero-release
     mass balance, per week, per dam, and pooled.

Usage:
    python scripts/eval_levels_rulecurve.py \
        --pred-dir <dir with predictions_{val,test}.npz> \
        --out outputs/level_rulecurve
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WRIS = PROJECT_ROOT / "data" / "raw" / "wris_v2"
TRAIN_END = pd.Timestamp("2022-12-31")
TMC_PER_M3SDAY = 86400.0 * 35.3146667 / 1e9
MCM_PER_TMC = 28.3168466


def pooled_nse(obs, pred):
    obs, pred = np.asarray(obs, float).ravel(), np.asarray(pred, float).ravel()
    m = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[m], pred[m]
    d = ((obs - obs.mean()) ** 2).sum()
    return float(1 - ((pred - obs) ** 2).sum() / d) if d > 0 else float("nan")


def implied_releases(storage_tmc: pd.Series, inflow_m3s: pd.Series) -> pd.Series:
    """R(t) = S(t) + I(t) - S(t+1), in TMC/day -> TMC per day, clipped >= 0."""
    i_tmc = inflow_m3s.astype(float) * TMC_PER_M3SDAY           # m3/s-day -> TMC
    r = storage_tmc.astype(float) + i_tmc - storage_tmc.shift(-1).astype(float)
    return r.clip(lower=0.0)


def build_rulecurve(storage: pd.Series, releases: pd.Series, cap: float, seed=42):
    """Fit R_hat = f(storage_fraction, sin/cos(doy)) on training rows only."""
    from sklearn.ensemble import GradientBoostingRegressor
    df = pd.DataFrame({"s": storage, "r": releases}).dropna()
    df = df[df.index <= TRAIN_END]
    if len(df) < 200:
        return None
    frac = (df["s"] / cap).values
    doy = df.index.dayofyear.values.astype(float)
    X = np.column_stack([frac, np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)])
    y = df["r"].values
    gb = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                   random_state=seed)
    gb.fit(X, y)
    return gb


def predict_release(gb, s_frac: float, t: pd.Timestamp) -> float:
    doy = float(t.dayofyear)
    X = np.array([[s_frac, np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25)]])
    return max(0.0, float(gb.predict(X)[0]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out", default="outputs/level_rulecurve")
    args = ap.parse_args()
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    daily = {}
    for f in sorted(WRIS.glob("*.csv")):
        df = pd.read_csv(f, parse_dates=["Date"]).set_index("Date").sort_index()
        daily[f.stem] = {"inflow": df["Inflow (cusecs/cumecs)"].astype(float),
                         "storage": df["Storage (TMC/MCM)"].astype(float)}

    rows = []
    for split in ("val", "test"):
        p = pred_dir / f"predictions_{split}.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        names = [str(r) for r in z["reservoirs"]]
        slugs = [n.lower().replace(" ", "_") for n in names]
        gnn = z["targets"]                                   # (S, N, 12) TMC-scaled? -> m3/s-days
        dates = pd.to_datetime(pd.Series([str(d) for d in z["dates"]]))

        for j, s in enumerate(slugs):
            st = daily[s]["storage"]; cap = float(st.max())
            rel = implied_releases(st, daily[s]["inflow"])
            gb = build_rulecurve(st, rel, cap)
            if gb is None:
                continue
            clim_by_doy = st[st.index <= TRAIN_END].groupby(st[st.index <= TRAIN_END].index.dayofyear).mean()
            for i in range(len(dates)):
                t = dates.iloc[i]
                if t + pd.Timedelta(days=84) > st.index.max():
                    continue
                s0 = float(np.clip(st.asof(t), 0, cap))
                s_rc = s_zero = s_clim = s_pers = s0
                for w in range(1, 13):
                    i_mcm = float(gnn[i, j, w - 1])                  # m3/s-days
                    i_tmc = i_mcm * TMC_PER_M3SDAY
                    tw = t + pd.Timedelta(days=7 * w)
                    # rule-curve release
                    r_hat = predict_release(gb, max(s_rc / cap, 0.0), tw)
                    s_rc = float(np.clip(s_rc + i_tmc - r_hat, 0, cap))
                    # zero-release mass balance (upper bound on storage)
                    s_zero = float(np.clip(s_zero + i_tmc, 0, cap))
                    # seasonal climatology of storage
                    sdoy = int(tw.dayofyear)
                    s_clim = float(clim_by_doy.get(sdoy, s0))
                    rows.append({"Split": split, "Reservoir": names[j], "Week": w,
                                 "_act": float(st.asof(tw)), "_rc": s_rc,
                                 "_zero": s_zero, "_clim": s_clim, "_pers": s_pers})

    raw = pd.DataFrame(rows)
    agg = []
    for keys, g in raw.groupby(["Split", "Week"]):
        rec = {"Split": keys[0], "Week": int(keys[1]), "n": len(g)}
        for tag, col in [("rulecurve", "_rc"), ("zero_release", "_zero"),
                         ("climatology", "_clim"), ("persistence", "_pers")]:
            rec[f"NSE_{tag}"] = round(pooled_nse(g["_act"], g[col]), 3)
        agg.append(rec)
    agg = pd.DataFrame(agg).sort_values(["Split", "Week"])
    agg.to_csv(out_dir / "level_rulecurve_pooled.csv", index=False)

    per_dam = []
    for keys, g in raw.groupby(["Split", "Reservoir", "Week"]):
        per_dam.append({"Split": keys[0], "Reservoir": keys[1], "Week": int(keys[2]),
                        "NSE_rulecurve": round(pooled_nse(g["_act"], g["_rc"]), 3),
                        "NSE_persistence": round(pooled_nse(g["_act"], g["_pers"]), 3)})
    per_dam = pd.DataFrame(per_dam)
    per_dam.to_csv(out_dir / "level_rulecurve_per_dam.csv", index=False)

    pd.set_option("display.width", 200)
    print("\n" + "=" * 88)
    print("SELF-CONTAINED LEVEL (rule-curve release, no future knowledge) -- pooled NSE")
    print("=" * 88)
    print(agg.to_string(index=False))
    print("\nweek-1, per dam:")
    print(per_dam[per_dam.Week == 1].sort_values(["Split", "NSE_rulecurve"], ascending=[True, False]).to_string(index=False))
    print(f"\nSaved -> {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
