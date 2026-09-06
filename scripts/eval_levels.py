"""Stage-2 evaluation: reservoir STORAGE/LEVEL forecasting from GNN inflow predictions.

Water balance per forecast week (all volumes in TMC):
    S(t+w) = S(t) + sum_k( I_k - R_k )
  - I_k  : GNN P50 weekly inflow predictions (converted m3/s-days -> TMC)
  - R_k  : releases, modeled per dam from its OWN operational history
           (median observed release by calendar month, fitted on train years)
  - S(t) : cleaned storage at the sample date (wris_v2)

Baselines: storage persistence (S(t) held flat) and seasonal storage
climatology (doy-mean from train years). Metrics: NSE per reservoir/week.

Usage: python scripts/eval_levels.py [--pred-dir outputs/run7/seed42]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"C:\Users\odeda\Desktop\Projects\PBL")
WRIS = PROJECT_ROOT / "data" / "raw" / "wris_v2"
TRAIN_END = pd.Timestamp("2022-12-31")
TMC_PER_M3SDAY = 86400.0 * 35.3146667 / 1e9  # m3/s-day -> TMC (0.0030512)


def nse(o: np.ndarray, p: np.ndarray) -> float:
    d = ((o - o.mean()) ** 2).sum()
    return float(1 - ((p - o) ** 2).sum() / d) if d > 0 else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", default="outputs/run7/seed42")
    parser.add_argument("--out", default="outputs/run7/storage_levels_summary.csv")
    args = parser.parse_args()

    pred_dir = Path(args.pred_dir)
    val = np.load(pred_dir / "predictions_val.npz", allow_pickle=True)
    test = np.load(pred_dir / "predictions_test.npz", allow_pickle=True)
    reservoirs = [str(r) for r in val["reservoirs"]]
    slugs = [r.lower().replace(" ", "_") for r in reservoirs]

    daily = {}
    for s in slugs:
        df = pd.read_csv(WRIS / f"{s}.csv", parse_dates=["Date"]).set_index("Date").sort_index()
        daily[s] = {
            "inflow": df["Inflow (cusecs/cumecs)"].astype(float),
            "storage": df["Storage (TMC/MCM)"].astype(float),
        }

    # observed weekly inflow volumes in TMC and weekly storage steps
    def week_volume(s: str, t: pd.Timestamp, w: int) -> float:
        win = daily[s]["inflow"].loc[t + pd.Timedelta(days=7 * (w - 1) + 1): t + pd.Timedelta(days=7 * w)]
        return float(win.sum() * TMC_PER_M3SDAY)

    def storage_at(s: str, t: pd.Timestamp) -> float:
        return float(daily[s]["storage"].asof(t))

    # release model: median observed monthly release (TMC/week), train years
    release_tables = {}
    for s in slugs:
        rows = []
        for t in pd.date_range("2010-07-01", TRAIN_END - pd.Timedelta(days=84), freq="D"):
            for w in (1,):
                i_vol = week_volume(s, t, 1)
                s0, s1 = storage_at(s, t), storage_at(s, t + pd.Timedelta(days=7))
                r = s0 + i_vol - s1  # implied release (TMC/week)
                if 0 <= r <= max(0.5, 0.9 * (daily[s]["storage"].max() - daily[s]["storage"].min())):
                    rows.append((t.month, r))
        rt = pd.DataFrame(rows, columns=["month", "r"])
        release_tables[s] = rt.groupby("month")["r"].median()
    print("release climatology fitted on train years (median TMC/week by month)")

    # persistence storage baseline: S held flat
    rows = []
    for split_name, npz, dates in [
        ("val", val, pd.to_datetime(pd.Series([str(d) for d in val["dates"]]))),
        ("test", test, pd.to_datetime(pd.Series([str(d) for d in test["dates"]]))),
    ]:
        targets = npz["targets"]        # (S, N, 12) weekly volumes m3/s-days
        preds = npz["preds_median"]     # (S, N, 12)
        for j, (res, s) in enumerate(zip(reservoirs, slugs)):
            rt = release_tables[s]
            for i in range(len(dates)):
                t = dates.iloc[i]
                if t + pd.Timedelta(days=84) > daily[s]["storage"].index.max():
                    continue
                s0 = storage_at(s, t)
                s0_tmc = s0
                cum_i = cum_r = 0.0
                for w in range(1, 13):
                    cum_i += targets[i, j, w - 1] * TMC_PER_M3SDAY
                    m = int((t + pd.Timedelta(days=7 * w - 4)).month)
                    cum_r += float(rt.get(m, rt.median()))
                    s_pred = max(0.0, min(s0_tmc + cum_i - cum_r, daily[s]["storage"].max()))
                    s_act = storage_at(s, t + pd.Timedelta(days=7 * w))
                    s_persist = s0_tmc
                    # seasonal storage climatology
                    sdoy = int((t + pd.Timedelta(days=7 * w)).dayofyear)
                    tr_s = daily[s]["storage"].loc[:TRAIN_END]
                    clim_s = float(tr_s[tr_s.index.dayofyear == sdoy].mean()) if (tr_s.index.dayofyear == sdoy).any() else s0_tmc
                    rows.append({
                        "Split": split_name, "Reservoir": res, "Week": w, "Date": str(t.date()),
                        "NSE_s_gnn": nse(np.array([s_act]), np.array([s_pred])),  # placeholder, pooled later
                        "_s_act": s_act, "_s_gnn": s_pred, "_s_persist": s_persist, "_s_clim": clim_s,
                    })
    raw = pd.DataFrame(rows)

    # pooled NSE per (split, reservoir, week) over samples
    out_rows = []
    for (split, res, w), g in raw.groupby(["Split", "Reservoir", "Week"]):
        for tag, pcol in [("gnn", "_s_gnn"), ("persist", "_s_persist"), ("clim", "_s_clim")]:
            o, p = g["_s_act"].values, g[pcol].values
            d = ((o - o.mean()) ** 2).sum()
            nse_v = float(1 - ((p - o) ** 2).sum() / d) if d > 0 else float("nan")
            out_rows.append({"Split": split, "Reservoir": res, "Week": int(w), "Model": tag, "NSE": round(nse_v, 3),
                             "RMSE_TMC": round(float(np.sqrt(((p - o) ** 2).mean())), 3)})
    out = pd.DataFrame(out_rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)

    # console: test split, weeks 1/4/12, per reservoir
    print()
    print("=== RESERVOIR LEVEL (storage, TMC) FORECAST — held-out 2024 ===")
    print("GNN = inflow model routed through the dam's own release history")
    for w in (1, 4, 12):
        sub = out[(out["Split"] == "test") & (out["Week"] == w)].pivot(index="Reservoir", columns="Model", values="NSE")
        if sub.empty:
            continue
        print(f"\n-- week {w} --")
        print(sub[["gnn", "persist", "clim"]].round(3).to_string())
        gnn_wins_p = int((sub["gnn"] > sub["persist"]).sum())
        gnn_wins_c = int((sub["gnn"] > sub["clim"]).sum())
        print(f"   GNN > persistence: {gnn_wins_p}/{len(sub)} | GNN > climatology: {gnn_wins_c}/{len(sub)}")
    print(f"\nSaved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
