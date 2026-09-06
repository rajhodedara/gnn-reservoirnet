"""Stage-2 v2: reservoir storage forecasting with a learned release model.

v1 (scripts/eval_levels.py) routed GNN inflow predictions through a monthly
MEDIAN release climatology — too crude (irrigation demand varies year to
year). v2 fits, per dam on TRAIN years only, a ridge regression:

    R_week(t) ~ f(month, storage% of live, storage%^2, previous week inflow)

and routes GNN P50 inflow + fitted releases through the mass balance:

    S(t+w) = clip(S(t) + sum(I_pred_k - R_pred_k), 0, gross_cap_tmc)

Baselines: storage persistence, seasonal storage climatology, and v1's
monthly-median releases. Metrics: storage NSE per dam/week (held-out 2024).

Usage: python scripts/eval_levels_v2.py --pred-dir outputs/run7/seed42
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(r"C:\Users\odeda\Desktop\Projects\PBL")
WRIS = PROJECT_ROOT / "data" / "raw" / "wris_v2"
TRAIN_END = pd.Timestamp("2022-12-31")
TMC_PER_M3SDAY = 86400.0 * 35.3146667 / 1e9


def nse(o: np.ndarray, p: np.ndarray) -> float:
    d = ((o - o.mean()) ** 2).sum()
    return float(1 - ((p - o) ** 2).sum() / d) if d > 0 else float("nan")


def ridge_fit(X: np.ndarray, y: np.ndarray, lam: float = 1.0) -> np.ndarray:
    """Closed-form ridge with intercept; X should already be standardized."""
    A = np.hstack([X, np.ones((X.shape[0], 1))])
    A[abs(A) < 1e-12] = 1e-12
    return np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ y)


def month_onehot(months: np.ndarray) -> np.ndarray:
    M = np.zeros((len(months), 11))
    for i, m in enumerate(months):
        if m > 1:
            M[i, m - 2] = 1.0
    return M


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-dir", default="outputs/run7/seed42")
    parser.add_argument("--out", default="outputs/levels_v2_summary.csv")
    parser.add_argument("--lam", type=float, default=1.0)
    args = parser.parse_args()

    with open(PROJECT_ROOT / "configs" / "reservoirs.yaml") as f:
        res_cfg = yaml.safe_load(f)["reservoirs"]
    cap_mcm = {r["id"]: float(r.get("gross_capacity_mcm", r.get("gross_storage_mcm", 0))) for r in res_cfg}

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
    live_tmc = {s: cap_mcm[s] / 28.3168466 for s in slugs}

    def week_volume(s, t, w):
        win = daily[s]["inflow"].loc[t + pd.Timedelta(days=7 * (w - 1) + 1): t + pd.Timedelta(days=7 * w)]
        return float(win.sum() * TMC_PER_M3SDAY)

    def storage_at(s, t):
        return float(daily[s]["storage"].asof(t))

    models, v1_tables = {}, {}
    for s in slugs:
        fit_rows = []
        for t in pd.date_range("2010-07-01", TRAIN_END - pd.Timedelta(days=84), freq="D"):
            s0 = storage_at(s, t)
            s1 = storage_at(s, t + pd.Timedelta(days=7))
            i_vol = week_volume(s, t, 1)
            r = s0 + i_vol - s1
            spct = min(max(s0 / max(live_tmc[s], 1e-6), 0.0), 1.0)
            fit_rows.append((t, s0, spct, i_vol, r))
        tr = pd.DataFrame(fit_rows, columns=["t", "s0", "spct", "i_vol", "r"])
        q999 = tr["r"].quantile(0.999)
        tr = tr[(tr["r"] >= 0) & (tr["r"] <= q999)]
        months = tr["t"].dt.month.values
        X = np.hstack([
            month_onehot(months),
            tr["spct"].values[:, None], tr["spct"].values[:, None] ** 2,
            (tr["i_vol"].values / max(tr["i_vol"].quantile(0.95), 1e-6))[:, None],
        ])
        mu, sd = X.mean(axis=0), X.std(axis=0); sd[sd == 0] = 1.0
        Xs = (X - mu) / sd
        models[s] = {"w": ridge_fit(Xs, tr["r"].values, args.lam), "mu": mu, "sd": sd, "fallback": float(tr["r"].median())}
        # v1: monthly median table (for comparison)
        v1_tables[s] = tr.assign(month=tr["t"].dt.month).groupby("month")["r"].median()

    sample_rows = []
    for split_name, npz, dates in [
        ("val", val, pd.to_datetime(pd.Series([str(d) for d in val["dates"]]))),
        ("test", test, pd.to_datetime(pd.Series([str(d) for d in test["dates"]]))),
    ]:
        targets, preds = npz["targets"], npz["preds_median"]
        for j, (res, s) in enumerate(zip(reservoirs, slugs)):
            mdl, rt = models[s], v1_tables[s]
            cap_tmc = daily[s]["storage"].max()
            for i in range(len(dates)):
                t = dates.iloc[i]
                if t + pd.Timedelta(days=84) > daily[s]["storage"].index.max():
                    continue
                s0 = storage_at(s, t)
                s0 = min(max(s0, 0.0), cap_tmc)
                cum_i = cum_r = cum_r_v2 = 0.0
                s_pred_v2, s_pred_v1 = s0, s0
                for w in range(1, 13):
                    i_tmc = targets[i, j, w - 1] * TMC_PER_M3SDAY
                    cum_i += i_tmc
                    tt = t + pd.Timedelta(days=7 * w - 4)
                    spct = min(max(s0 / max(live_tmc[s], 1e-6), 0.0), 1.0)
                    iv_n = min(preds[i, j, w - 1] / max(1e-6, 1.0), 5.0)
                    feats = np.hstack([month_onehot(np.array([tt.month])), [[spct, spct ** 2, iv_n]]])
                    feats_s = (feats - mdl["mu"]) / mdl["sd"]
                    feats_s[abs(feats_s) < 1e-12] = 1e-12
                    r_v2 = float(np.clip(feats_s @ mdl["w"][: -1] + mdl["w"][-1], 0.0, None))
                    cum_r_v2 += r_v2
                    s_pred_v2 = float(np.clip(s0 + cum_i - cum_r_v2, 0.0, cap_tmc))
                    m = int(tt.month)
                    cum_r += float(rt.get(m, rt.median()))
                    s_pred_v1 = float(np.clip(s0 + cum_i - cum_r, 0.0, cap_tmc))
                    s_act = storage_at(s, t + pd.Timedelta(days=7 * w))
                    s_persist = s0
                    sdoy = int((t + pd.Timedelta(days=7 * w)).dayofyear)
                    tr_s = daily[s]["storage"].loc[:TRAIN_END]
                    clim_s = float(tr_s[tr_s.index.dayofyear == sdoy].mean()) if (tr_s.index.dayofyear == sdoy).any() else s0
                    sample_rows.append({"Split": split_name, "Reservoir": res, "Week": w, "_s_act": s_act,
                                 "_s_v2": s_pred_v2, "_s_v1": s_pred_v1, "_s_persist": s_persist, "_s_clim": clim_s})
    raw = pd.DataFrame(sample_rows)
    out_rows2 = []
    for (split, res, w), g in raw.groupby(["Split", "Reservoir", "Week"]):
        o = g["_s_act"].values
        for tag, pc in [("v2_ridge", "_s_v2"), ("v1_median", "_s_v1"), ("persist", "_s_persist"), ("clim", "_s_clim")]:
            p = g[pc].values
            d = ((o - o.mean()) ** 2).sum()
            nse_v = float(1 - ((p - o) ** 2).sum() / d) if d > 0 else float("nan")
            out_rows2.append({"Split": split, "Reservoir": res, "Week": int(w), "Model": tag,
                              "NSE": round(nse_v, 3), "RMSE_TMC": round(float(np.sqrt(((p - o) ** 2).mean())), 2)})
    res_df = pd.DataFrame(out_rows2)
    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(outp, index=False)

    print()
    print("=== STAGE-2 v2 vs v1 vs baselines — storage NSE, held-out 2024 ===")
    for w in (1, 4, 12):
        sub = res_df[(res_df["Split"] == "test") & (res_df["Week"] == w)].pivot(index="Reservoir", columns="Model", values="NSE")
        if sub.empty:
            continue
        sub = sub[["v2_ridge", "v1_median", "persist", "clim"]]
        print(f"\n-- week {w} --")
        print(sub.round(3).sort_values("v2_ridge", ascending=False).to_string())
        print(f"   means: v2={sub['v2_ridge'].mean():.3f}  v1={sub['v1_median'].mean():.3f}  persist={sub['persist'].mean():.3f}  clim={sub['clim'].mean():.3f}")
        print(f"   v2 > persist: {(sub['v2_ridge'] > sub['persist']).sum()}/{len(sub)} | v2 > v1: {(sub['v2_ridge'] > sub['v1_median']).sum()}/{len(sub)}")
    print(f"\nSaved: {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
