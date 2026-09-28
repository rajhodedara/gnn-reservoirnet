#!/usr/bin/env python
"""Analyse a set of rolling-origin folds by climate phase.

Reads `outputs/rolling_origin/fold_<year>/` (produced by rolling_origin_eval.py)
plus the ONI series, and reports week-1..12 inflow NSE tagged with each fold's
climate phase (El Nino / La Nina / neutral by JJAS mean ONI). This is what turns
"we validated on 2015-16 and 2023-24" into an evidence-backed statement.

Usage:
    python scripts/analyse_el_nino_folds.py --rolling-dir outputs/rolling_origin \
        --out outputs/el_nino/folds_by_phase.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENSO = PROJECT_ROOT / "data" / "raw" / "enso" / "combined_climate_indices.csv"
JJAS = {6, 7, 8, 9}


def pooled_nse(obs, pred):
    obs, pred = np.asarray(obs, float).ravel(), np.asarray(pred, float).ravel()
    m = np.isfinite(obs) & np.isfinite(pred)
    obs, pred = obs[m], pred[m]
    d = ((obs - obs.mean()) ** 2).sum()
    return float(1 - ((pred - obs) ** 2).sum() / d) if d > 0 else float("nan")


def phase_for_year(oni: pd.Series, year: int) -> tuple:
    y = oni[(oni.index.year == year) & (oni.index.month.isin(JJAS))]
    ref = float(y.mean()) if len(y) else float("nan")
    if ref >= 0.5:
        p = "El Nino"
    elif ref <= -0.5:
        p = "La Nina"
    else:
        p = "neutral"
    return p, ref


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rolling-dir", default="outputs/rolling_origin")
    ap.add_argument("--out", default="outputs/el_nino/folds_by_phase.csv")
    args = ap.parse_args()

    rd = PROJECT_ROOT / args.rolling_dir
    oni = pd.read_csv(ENSO, parse_dates=["Date"]).set_index("Date").resample("D").ffill()["oni"]

    rows = []
    for fd in sorted(rd.glob("fold_*")):
        year = int(fd.name.split("_")[1])
        ph, ref = phase_for_year(oni, year)
        row = {"fold": year, "jjas_oni": round(ref, 2), "phase": ph}
        # prefer pooled-by-week if present, else per-reservoir mean
        pw = fd / "evaluation_metrics_by_week_test.csv"
        pr = fd / "evaluation_metrics_per_reservoir_test.csv"
        if pw.exists():
            d = pd.read_csv(pw)
            means = d.groupby("Week")["NSE"].mean()
            for w, v in means.items():
                row[f"week_{int(w)}"] = round(float(v), 3)
        if pr.exists():
            d = pd.read_csv(pr)
            row["mean_NSE"] = round(float(d["NSE"].mean()), 3)
        # cross-check against any stored npz for a pooled number
        npz = fd / "predictions_test.npz"
        if npz.exists():
            z = np.load(npz, allow_pickle=True)
            t, p = z["targets"][:, :, 0], z["preds_median"][:, :, 0]
            row["pooled_week1_NSE"] = round(pooled_nse(t, p), 3)
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["phase", "fold"])
    out = PROJECT_ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    pd.set_option("display.width", 220)
    print("=" * 96)
    print("ROLLING-ORIGIN FOLDS BY CLIMATE PHASE (JJAS mean ONI)")
    print("=" * 96)
    cols = [c for c in ["fold", "phase", "jjas_oni", "week_1", "week_2", "week_4",
                        "week_8", "week_12", "pooled_week1_NSE", "mean_NSE"] if c in df.columns]
    print(df[cols].to_string(index=False))

    if "pooled_week1_NSE" in df.columns:
        print("\nWeek-1 pooled NSE by phase:")
        print(df.groupby("phase")["pooled_week1_NSE"].agg(["mean", "min", "max", "count"]).round(3).to_string())
    print(f"\nSaved: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
