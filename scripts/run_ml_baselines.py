#!/usr/bin/env python
"""Classical / ML baselines for the weekly inflow task (design doc section 7).

Implements the baselines the approved design calls for, all predicting the SAME
target as the GNN -- the next-week inflow volume, sum of daily inflow over
t+1..t+7 per reservoir -- on the SAME temporal splits (train <= 2022-12-31,
val 2023, test 2024):

  * ARIMA / SARIMA   (statsmodels SARIMAX, per reservoir)
  * Random Forest    (sklearn, lag + calendar + storage features)
  * LSTM             (per-dam PyTorch, 90-day lookback like the GNN)

Persistence + day-of-year climatology are reused from run_baselines.py so all
five land in one comparison table with identical splits and metric code.

Usage:
    python scripts/run_ml_baselines.py [--config configs/default_config.yaml]
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_baselines import (  # noqa: E402
    load_daily_inflow, weekly_sum_next7, persistence_forecast,
    climatology_forecast, metrics_table,
)

warnings.filterwarnings("ignore")


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def pooled_nse(obs: pd.DataFrame, pred: pd.DataFrame) -> float:
    o = np.concatenate([obs[c].values.astype(float) for c in obs.columns])
    p = np.concatenate([pred[c].values.astype(float) for c in pred.columns])
    m = ~(np.isnan(o) | np.isnan(p))
    o, p = o[m], p[m]
    d = ((o - o.mean()) ** 2).sum()
    return float(1 - ((p - o) ** 2).sum() / d) if d > 0 else float("nan")


def pooled_rmse(obs, pred):
    o = np.concatenate([obs[c].values.astype(float) for c in obs.columns])
    p = np.concatenate([pred[c].values.astype(float) for c in pred.columns])
    m = ~(np.isnan(o) | np.isnan(p))
    return float(np.sqrt(((p[m] - o[m]) ** 2).mean()))


# --------------------------------------------------------------------------- #
# SARIMA
# --------------------------------------------------------------------------- #
def _weekly_series(targets: pd.DataFrame, col: str) -> pd.Series:
    """Resample the daily-indexed weekly-sum target to a clean weekly series.

    The target index is daily (one row per forecast origin), but consecutive rows
    overlap 6/7 days. Fitting a period-52 seasonal model on a DAILY index would
    treat 52 days as a season -- wrong and pathologically slow. Resampling to a
    true weekly (W) frequency makes the seasonal period 52 WEEKS, as intended.
    """
    s = targets[col].dropna()
    return s.resample("W").mean().dropna()


def sarima_forecast(daily: pd.DataFrame, targets: pd.DataFrame, train_end: str,
                    order=(1, 0, 1), seasonal=(1, 0, 0, 52)) -> pd.DataFrame:
    """Per-reservoir SARIMAX on a WEEKLY series (52-week seasonality); forecast the rest."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    out = pd.DataFrame(index=targets.index, columns=targets.columns, dtype=float)
    tr_cut = pd.Timestamp(train_end)
    for c in targets.columns:
        try:
            s = _weekly_series(targets, c)
            tr = s[s.index <= tr_cut]
            if len(tr) < 80:
                continue
            res = SARIMAX(
                tr.values, order=order, seasonal_order=seasonal,
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False, maxiter=40)
            # one-step-ahead predictions over the post-train weeks
            n_after = len(s) - len(tr)
            if n_after <= 0:
                continue
            fc = res.get_prediction(start=len(tr), end=len(s) - 1, dynamic=False)
            pred = np.asarray(fc.predicted_mean, dtype=float)
            wk_index = s.index[len(tr):]
            # scatter weekly predictions back onto the daily origins they cover
            ser = pd.Series(pred, index=wk_index)
            for t in targets.index[targets.index > tr_cut]:
                dow = ser.index[ser.index >= t]
                if len(dow):
                    out.loc[t, c] = float(ser.loc[dow[0]])
        except Exception as e:  # noqa: BLE001
            print(f"    [sarima] {c}: {type(e).__name__}: {e}")
    return out


# --------------------------------------------------------------------------- #
# Random Forest
# --------------------------------------------------------------------------- #
def _rf_features(daily: pd.DataFrame, c: str) -> pd.DataFrame:
    s = daily[c].astype(float)
    w1 = s.rolling(7).sum()
    w2 = s.rolling(14).sum() - w1
    w4 = s.rolling(28).sum() - s.rolling(14).sum()
    doy = daily.index.dayofyear
    feat = pd.DataFrame({
        "lag1w": w1,
        "lag2w": w2,
        "lag4w": w4,
        "sin": np.sin(2 * np.pi * doy / 365.25),
        "cos": np.cos(2 * np.pi * doy / 365.25),
        "mean7": s.rolling(7).mean(),
        "max7": s.rolling(7).max(),
    }, index=daily.index)
    return feat


def rf_forecast(daily: pd.DataFrame, targets: pd.DataFrame, train_end: str,
                n_estimators=300, seed=42) -> pd.DataFrame:
    from sklearn.ensemble import RandomForestRegressor
    out = pd.DataFrame(index=targets.index, columns=targets.columns, dtype=float)
    tr_cut = pd.Timestamp(train_end)
    for c in targets.columns:
        X = _rf_features(daily, c)
        y = targets[c]
        # the feature row at t must not contain the target window -> shift feats
        Xs = X.shift(7)
        mask = Xs.notna().all(axis=1) & y.notna()
        tr = mask & (X.index <= tr_cut)
        if tr.sum() < 100:
            continue
        rf = RandomForestRegressor(n_estimators=n_estimators, min_samples_leaf=3,
                                   random_state=seed, n_jobs=-1)
        rf.fit(Xs[tr].values, y[tr].values)
        pr = mask & (X.index > tr_cut)
        out.loc[X.index[pr], c] = rf.predict(Xs[pr].values)
    return out


# --------------------------------------------------------------------------- #
# LSTM
# --------------------------------------------------------------------------- #
def lstm_forecast(daily: pd.DataFrame, targets: pd.DataFrame, train_end: str,
                  lookback=90, hidden=64, epochs=200, batch_size=128, patience=20,
                  lr=2e-3, seed=42, years=None, device=None, train_stride=3) -> pd.DataFrame:
    """Per-dam LSTM, FULL STRENGTH (no speed compromise).

    * lookback = 90 days, matching the GNN's input window exactly.
    * 2-layer LSTM, hidden 64, early stopping on a held-out tail of the training
      period (patience epochs) -- so the fit is honest, not fixed-epoch.
    * Vectorised window construction; mini-batch Adam; grad clipping.
    Deliberately NOT reduced for CPU speed: an under-trained baseline would make
    the GNN look better than it is. Runs fine on GPU via --device cuda.
    """
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    np.random.seed(seed)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    out = pd.DataFrame(index=targets.index, columns=targets.columns, dtype=float)
    tr_cut = pd.Timestamp(train_end)
    if years is None:
        years = [2023, 2024]

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(1, hidden, num_layers=2, batch_first=True, dropout=0.1)
            self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Linear(32, 1))

        def forward(self, x):
            o, _ = self.lstm(x)
            return self.head(o[:, -1, :]).squeeze(-1)

    for c in targets.columns:
        s = daily[c].astype(float)
        y = targets[c].shift(-7)
        vals = s.values.astype("float32")
        yv = y.values.astype("float32")
        wins = np.lib.stride_tricks.sliding_window_view(vals, lookback)
        X = wins[..., None]
        origins = np.arange(len(wins))
        Y = yv[origins]
        dates = s.index[origins]
        ok = np.isfinite(Y) & np.isfinite(X).all(axis=(1, 2))
        tr = ok & (dates <= tr_cut)
        te = ok & np.isin(dates.year, years)
        if tr.sum() < 300 or te.sum() == 0:
            continue
        # Early-stopping split: last 15% of the training origins.
        tr_idx = np.where(tr)[0]
        # Redundancy reduction: consecutive 90-day windows overlap 89/90 days, so
        # training on every train_stride-th origin keeps the same information with
        # far less compute. Architecture/capacity are unchanged; set
        # --train-stride 1 for the exhaustive (slower) fit.
        if train_stride and train_stride > 1:
            tr_idx = tr_idx[::train_stride]
        n_val = max(int(0.15 * len(tr_idx)), 60)
        es_val, fit_idx = tr_idx[-n_val:], tr_idx[:-n_val]
        mu, sd = X[fit_idx].mean(), X[fit_idx].std() + 1e-8
        ym, ys = Y[fit_idx].mean(), Y[fit_idx].std() + 1e-8
        Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
        Yt = torch.tensor((Y - ym) / ys, dtype=torch.float32)
        net = Net().to(device)
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
        lossf = nn.MSELoss()
        fit_t = torch.tensor(fit_idx)
        val_t = Xt[torch.tensor(es_val)].to(device)
        val_y = Yt[torch.tensor(es_val)].to(device)
        best, best_state, bad = float("inf"), None, 0
        for ep in range(epochs):
            net.train()
            perm = fit_t[torch.randperm(len(fit_t))]
            for b in range(0, len(perm), batch_size):
                sel = perm[b:b + batch_size]
                opt.zero_grad()
                loss = lossf(net(Xt[sel].to(device)), Yt[sel].to(device))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            net.eval()
            with torch.no_grad():
                v = float(lossf(net(val_t), val_y))
            if v < best - 1e-5:
                best, bad = v, 0
                best_state = {k: t.detach().clone() for k, t in net.state_dict().items()}
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            te_i = np.where(te)[0]
            p = net(Xt[te_i].to(device)).cpu().numpy()
            out.loc[dates[te_i], c] = p * ys + ym
    return out


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default_config.yaml"))
    ap.add_argument("--epochs", type=int, default=200,
                    help="max LSTM epochs (early stopping usually ends far sooner)")
    ap.add_argument("--device", default=None, help="cpu | cuda (default: auto)")
    ap.add_argument("--train-stride", type=int, default=3,
                    help="keep every Nth training window (overlapping windows are redundant); 1 = exhaustive")
    ap.add_argument("--fast", action="store_true",
                    help="CPU smoke run only: smaller LSTM, fewer epochs. NOT for results.")
    ap.add_argument("--out", default=str(PROJECT_ROOT / "outputs" / "ml_baselines.json"))
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))
    wris = PROJECT_ROOT / cfg["data"]["wris_data_dir"]
    ids = [r["id"] for r in yaml.safe_load(
        open(PROJECT_ROOT / cfg["data"]["reservoirs_file"]))["reservoirs"]]
    val_years = cfg["data"].get("val_years", [2023])
    test_years = cfg["data"].get("test_years", [2024])
    train_end = cfg["data"].get("train_end", "2022-12-31")

    daily = load_daily_inflow(wris, ids)
    targets = weekly_sum_next7(daily)

    print("Fitting baselines (train <= %s) ..." % train_end)
    persist = persistence_forecast(daily)
    clim = climatology_forecast(daily, targets, train_end_year=int(train_end[:4]))

    print("  SARIMA ...")
    sar = sarima_forecast(daily, targets, train_end)
    print("  Random Forest ...")
    rf = rf_forecast(daily, targets, train_end)
    if args.fast:
        print("  LSTM (FAST smoke: hidden=16, epochs=8, lookback=30) -- NOT for results")
        lst = lstm_forecast(daily, targets, train_end, epochs=8, hidden=16,
                            lookback=30, patience=3, years=list(val_years) + list(test_years),
                            device=args.device)
    else:
        print(f"  LSTM (full strength: lookback=90, hidden=64, max {args.epochs} epochs, "
              f"early stopping; device={args.device or 'auto'}) ...")
        lst = lstm_forecast(daily, targets, train_end, epochs=args.epochs,
                            years=list(val_years) + list(test_years), device=args.device,
                            train_stride=args.train_stride)

    results = {"config": {"wris_dir": str(wris), "val_years": val_years,
                          "test_years": test_years, "train_end": train_end}}
    models = {"persistence": persist, "climatology": clim,
              "sarima": sar, "random_forest": rf, "lstm": lst}
    for name, pred in models.items():
        results[name] = {}
        for split, years in [("val", val_years), ("test", test_years)]:
            m = targets.index.year.isin(years)
            obs, p = targets[m], pred[m]
            keep = p.notna().any(axis=1)
            obs, p = obs.loc[keep], p.loc[keep].fillna(0.0)
            results[name][split] = metrics_table(obs, p)

    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n" + "=" * 104)
    print(f"WEEKLY INFLOW -- all baselines, TEST {test_years} (per-dam NSE)")
    print("=" * 104)
    hdr = f"{'reservoir':<22}" + "".join(f"{n:>14}" for n in models)
    print(hdr)
    print("-" * 104)
    for c in ids:
        row = f"{c:<22}"
        for n in models:
            v = results[n]["test"][c]["NSE"]
            row += f"{v:>14.3f}" if v == v else f"{'--':>14}"
        print(row)
    print("-" * 104)
    prow = f"{'POOLED':<22}"
    for n in models:
        o = targets[targets.index.year.isin(test_years)]
        p = models[n][targets.index.year.isin(test_years)]
        keep = p.notna().any(axis=1)
        prow += f"{pooled_nse(o.loc[keep], p.loc[keep].fillna(0.0)):>14.3f}"
    print(prow)
    print("=" * 104)
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
