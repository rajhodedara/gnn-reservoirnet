"""Per-week optimal blending of GNN predictions with seasonal climatology.

Motivation: the held-out horizon race shows the GNN wins weeks 1-2 while
seasonal climatology wins weeks 4+. This script fits, per (week, reservoir),
the blend weight alpha on the VALIDATION year that minimizes squared error of

    pred = alpha * gnn + (1 - alpha) * climatology

then applies the fitted weights to the held-out TEST year and reports the
resulting NSE curve against GNN-only and climatology-only.

Requires runs/predictions_val.npz and runs/predictions_test.npz (written by
main.py::evaluate). Works with a flat runs/ layout or per-seed directories.

Usage (repo root):
    python scripts/blend_eval.py [--runs-dir runs] [--seed-dir seed42]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_daily(wris_dir: Path, reservoirs):
    daily = {}
    for r in reservoirs:
        f = wris_dir / f"{r}.csv"
        slug = r.lower().replace(" ", "_")
        f = wris_dir / f"{slug}.csv"
        df = pd.read_csv(f, parse_dates=["Date"])
        daily[r] = df.set_index("Date")["Inflow (cusecs/cumecs)"].astype(float)
    return daily


def climatology_weekly(daily_single: pd.Series, sample_dates, week: int, train_end: str, window: int = 7):
    """Day-of-year climatology of the week-w inflow volume, from train years only.

    For sample end date t, the week-w target covers t+7(w-1)+1 .. t+7w; the
    climatology is indexed by the day-of-year of that window's midpoint.
    """
    lag = 7 * (week - 1)
    shifted = daily_single.shift(-(7 * week)).rolling(window).sum()  # at t: sum(t+7w-6..t+7w)
    obs = shifted.copy()
    tr = obs[obs.index <= pd.Timestamp(train_end) - pd.Timedelta(days=7 * week)]
    doy = tr.index.dayofyear.values
    vals = tr.values
    slots = np.arange(1, 367)
    dist = np.abs(slots[:, None] - doy[None, :])
    dist = np.minimum(dist, 366 - dist)
    in_win = dist <= 7
    sums = in_win @ np.nan_to_num(vals)
    counts = in_win.sum(axis=1)
    means = np.divide(sums, counts[:, None], out=np.full_like(sums, np.nan), where=counts[:, None] > 0)
    clim_doy = pd.Series(means, index=slots)

    mid_doy = (sample_dates + pd.Timedelta(days=7 * week - 3)).dayofyear.values
    pred = clim_doy.reindex(np.clip(mid_doy, 1, 366)).values
    fallback = np.nanmean(vals) if len(vals) else 0.0
    pred = np.where(np.isnan(pred), fallback, pred)
    return pred


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--seed-dir", default=None, help="e.g. seed42; default: flat runs/ or first seed dir")
    parser.add_argument("--wris-dir", default=str(PROJECT_ROOT / "data" / "raw" / "wris_v2"))
    args = parser.parse_args()

    runs_dir = Path(args.runs_dir)
    npz_dir = runs_dir
    if args.seed_dir:
        npz_dir = runs_dir / args.seed_dir
    elif not (runs_dir / "predictions_val.npz").exists():
        seed_dirs = sorted(d for d in runs_dir.glob("seed*") if (d / "predictions_val.npz").exists())
        if not seed_dirs:
            print("No predictions_*.npz found. Run training (main.py) first.")
            return 1
        npz_dir = seed_dirs[0]
    print(f"Using predictions from: {npz_dir}")

    val = np.load(npz_dir / "predictions_val.npz", allow_pickle=True)
    test = np.load(npz_dir / "predictions_test.npz", allow_pickle=True)
    reservoirs = [str(r) for r in val["reservoirs"]]
    val_dates = pd.to_datetime(pd.Series([str(d) for d in val["dates"]]))
    test_dates = pd.to_datetime(pd.Series([str(d) for d in test["dates"]]))

    wris_dir = Path(args.wris_dir)
    daily = load_daily(wris_dir, reservoirs)
    slug_map = {r: r.lower().replace(" ", "_") for r in reservoirs}

    num_weeks = val["targets"].shape[2]
    val_targets = val["targets"]        # (S, N, 12) raw weekly volumes
    val_gnn = val["preds_median"]
    test_targets = test["targets"]
    test_gnn = test["preds_median"]

    per_week = []
    weights_rows = []
    for w in range(1, num_weeks + 1):
        clim_v = np.column_stack([
            climatology_weekly(daily[r], val_dates, w, train_end="2022-12-31") for r in reservoirs
        ])
        clim_t = np.column_stack([
            climatology_weekly(daily[r], test_dates, w, train_end="2022-12-31") for r in reservoirs
        ])
        for j, r in enumerate(reservoirs):
            o_v = val_targets[:, j, w - 1]
            g_v = val_gnn[:, j, w - 1]
            c_v = clim_v[:, j]
            ok = ~(np.isnan(o_v) | np.isnan(g_v) | np.isnan(c_v))
            ov, gv, cv = o_v[ok], g_v[ok], c_v[ok]
            dc = gv - cv
            denom = (dc ** 2).sum()
            alpha = float(np.clip(((dc * (ov - cv)).sum() / denom) if denom > 0 else 0.0, 0.0, 1.0))

            o_t = test_targets[:, j, w - 1]
            g_t = test_gnn[:, j, w - 1]
            c_t = clim_t[:, j]
            okt = ~(np.isnan(o_t) | np.isnan(g_t) | np.isnan(c_t))
            ot, gt, ct = o_t[okt], g_t[okt], c_t[okt]

            def nse(o, p):
                dd = ((o - o.mean()) ** 2).sum()
                return float(1 - ((p - o) ** 2).sum() / dd) if dd > 0 else float("nan")

            blended_t = alpha * gt + (1 - alpha) * ct
            per_week.append({
                "Week": w, "Reservoir": r,
                "NSE_gnn": nse(ot, gt), "NSE_clim": nse(ot, ct), "NSE_blended": nse(ot, blended_t),
                "alpha": alpha,
            })
            weights_rows.append({"Week": w, "Reservoir": r, "alpha": round(alpha, 3)})

    df_all = pd.DataFrame(per_week)
    OUT = PROJECT_ROOT / "outputs"
    OUT.mkdir(exist_ok=True)
    df_all.to_csv(OUT / "blend_eval_full.csv", index=False)

    print()
    print("=== BLENDING VERDICT — held-out 2024, mean NSE across reservoirs ===")
    print(f"{'week':>4}{'GNN-only':>10}{'CLIM-only':>11}{'BLENDED':>9}")
    print("-" * 40)
    g = df_all.groupby("Week")[["NSE_gnn", "NSE_clim", "NSE_blended"]].mean()
    for w, r in g.iterrows():
        marker = " <-- GNN" if r["NSE_gnn"] >= r["NSE_clim"] else " <-- CLIM"
        print(f"{w:>4}{r['NSE_gnn']:>10.3f}{r['NSE_clim']:>11.3f}{r['NSE_blended']:>9.3f}{marker if r['NSE_blended'] >= max(r['NSE_gnn'], r['NSE_clim']) - 1e-9 else ''}")
    print("-" * 40)
    print(f"{'ALL':>4}{df_all['NSE_gnn'].mean():>10.3f}{df_all['NSE_clim'].mean():>11.3f}{df_all['NSE_blended'].mean():>9.3f}")
    print(f"\nSaved: {OUT / 'blend_eval_full.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
