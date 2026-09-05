"""Per-node predictability diagnostic: why do some reservoirs trail climatology?

Computes, on the held-out 2024 weekly inflow targets:
  - lag-1 autocorrelation (persistence headroom)
  - seasonal-climatology R^2 — variance explained by a "same week, average
    train-year" prediction (this is climatology's ceiling on 2024)
  - zero percentage
with the run #5 GNN NSE beside them.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WRIS = PROJECT_ROOT / "data" / "raw" / "wris_v2"
TRAIN_END = pd.Timestamp("2022-12-31")

slugs = ["almatti", "tungabhadra", "krishnaraja_sagara", "mettur", "nagarjuna_sagar",
         "srisailam", "jayakwadi", "ujjani", "sardar_sarovar", "ukai"]


def main() -> int:
    metrics_path = None
    for cand in sorted((PROJECT_ROOT / "outputs").glob("run*/seed*/evaluation_metrics_per_reservoir_test.csv")):
        metrics_path = cand
    met = {}
    if metrics_path:
        met = pd.read_csv(metrics_path).set_index("Reservoir")["NSE"]
        print(f"GNN NSE source: {metrics_path}\n")

    rows = []
    for slug in slugs:
        df = pd.read_csv(WRIS / f"{slug}.csv", parse_dates=["Date"]).set_index("Date")
        s = df["Inflow (cusecs/cumecs)"].astype(float)
        weekly_all = s.rolling(7).sum()

        # Train-year doy climatology of weekly volumes (2010-2022, 7-day buffer)
        tr = weekly_all.loc[: TRAIN_END - pd.Timedelta(days=14)].dropna()
        doy_tr = tr.index.dayofyear.values
        slots = np.arange(1, 367)
        dist = np.abs(slots[:, None] - doy_tr[None, :])
        dist = np.minimum(dist, 366 - dist)
        in_win = dist <= 7
        counts = in_win.sum(axis=1)
        clim_doy = pd.Series(
            np.divide(in_win @ np.nan_to_num(tr.values), counts,
                      out=np.full(366, np.nan), where=counts > 0),
            index=slots,
        )

        obs = weekly_all.loc["2024-01-01":"2024-12-31"].dropna()
        if len(obs) < 30 or obs.std() == 0:
            rows.append({"Reservoir": slug, "NSE_gnn": float("nan"), "lag1_ac": float("nan"),
                         "clim_R2": float("nan"), "zero_pct": 100.0})
            continue

        pred_clim = clim_doy.reindex(np.clip(obs.index.dayofyear, 1, 366)).values
        pred_clim = np.where(np.isnan(pred_clim), np.nanmean(clim_doy.values), pred_clim)
        ss_res = ((obs.values - pred_clim) ** 2).sum()
        ss_tot = ((obs.values - obs.mean()) ** 2).sum()
        clim_r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

        nse_val = met.get(slug.title().replace("_", " "), met.get(slug, np.nan))
        rows.append({
            "Reservoir": slug,
            "NSE_gnn": round(float(nse_val), 3) if nse_val == nse_val else float("nan"),
            "lag1_ac": round(float(obs.autocorr(lag=1)), 3),
            "clim_R2": round(clim_r2, 3),
            "zero_pct": round(float((weekly_all.loc["2024-01-01":"2024-12-31"] == 0).mean() * 100), 1),
        })

    out = pd.DataFrame(rows).sort_values("clim_R2", ascending=False)
    print(f"{'reservoir':<20}{'GNN NSE':>9}{'lag1_ac':>9}{'clim_R2':>9}{'zero%':>7}   reading")
    print("-" * 92)
    for _, r in out.iterrows():
        if r["clim_R2"] > 0.5 and (np.isnan(r["NSE_gnn"]) or r["NSE_gnn"] < r["clim_R2"]):
            reading = "seasonality-dominated: climatology is the bar"
        elif r["lag1_ac"] > 0.5:
            reading = "persistence-friendly: GNN should win here"
        else:
            reading = "low-predictability node"
        print(f"{r['Reservoir']:<20}{r['NSE_gnn']:>9.3f}{r['lag1_ac']:>9.3f}{r['clim_R2']:>9.3f}{r['zero_pct']:>7.1f}   {reading}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
