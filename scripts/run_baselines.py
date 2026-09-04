"""Weekly inflow-forecast baselines: weekly persistence + day-of-year climatology.

Mirrors the STGNN task (predict the next-7-day inflow volume per reservoir) so
the GNN's metrics are comparable. Uses data/raw/wris_v2/ and the same year
splits as configs/default_config.yaml (train < first val year; val = val_years;
test = test_years). No torch required.

Usage:
    python scripts/run_baselines.py [--config configs/default_config.yaml]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.wris_loader import load_reservoir_data  # noqa: E402


def load_daily_inflow(wris_dir: Path, reservoir_ids: list[str]) -> pd.DataFrame:
    """Daily inflow (m3/s) per reservoir, indexed by date, 2010-2024."""
    cols = {}
    for res_id in reservoir_ids:
        df = load_reservoir_data(str(wris_dir / f"{res_id}.csv"), res_id)
        if df.empty or "inflow" not in df.columns:
            raise FileNotFoundError(f"No inflow data for {res_id} in {wris_dir}")
        cols[res_id] = pd.to_numeric(df["inflow"], errors="coerce")
    daily = pd.DataFrame(cols).sort_index()
    daily = daily.loc["2010-01-01":"2024-12-31"]
    daily = daily.interpolate(method="time").ffill().bfill()
    if daily.isna().any().any():
        raise ValueError("NaN inflow after interpolation")
    return daily


def weekly_sum_next7(daily: pd.DataFrame) -> pd.DataFrame:
    """Target: sum of daily inflow over t+1..t+7 for each day t."""
    return daily.shift(-7).rolling(7).sum()


def persistence_forecast(daily: pd.DataFrame) -> pd.DataFrame:
    """Forecast: next-week volume = last-7-day volume ending at t."""
    return daily.rolling(7).sum()


def climatology_forecast(
    daily: pd.DataFrame,
    targets: pd.DataFrame,
    train_end_year: int,
    window_days: int = 7,
) -> pd.DataFrame:
    """Day-of-year climatology of weekly volumes, computed on training years only.

    For each day-of-year, the mean weekly target over training samples within
    +/- window_days (circular on 366 slots).
    """
    train_mask = targets.index <= pd.Timestamp(f"{train_end_year}-12-31") - pd.Timedelta(days=7)
    # Buffer excludes targets whose 7-day window crosses into val/test years.
    train_targets = targets[train_mask].dropna(how="all")

    doy = train_targets.index.dayofyear.values
    values = train_targets.values  # (n_train, nodes)
    slots = np.arange(1, 367)
    # Circular distance on a 366-day ring
    dist = np.abs(slots[:, None] - doy[None, :])
    dist = np.minimum(dist, 366 - dist)
    in_win = dist <= window_days  # (366, n_train)
    counts = in_win.sum(axis=1)
    sums = in_win @ np.nan_to_num(values)
    means = np.divide(sums, counts[:, None], out=np.full_like(sums, np.nan), where=counts[:, None] > 0)
    clim = pd.DataFrame(means, index=slots, columns=targets.columns)

    out_doy = targets.index.dayofyear.values
    pred = clim.loc[out_doy].values
    pred = np.where(np.isnan(pred), np.nanmean(values, axis=0), pred)  # fallback
    return pd.DataFrame(pred, index=targets.index, columns=targets.columns)


def metrics_table(obs: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """NSE / RMSE / MAE per reservoir plus pooled aggregate.

    Reservoirs whose observations are degenerate in the split (zero variance,
    e.g. all-zero gauge tails) are flagged and excluded from the pooled row.
    """
    rows = {}
    usable = []
    for col in obs.columns:
        o = obs[col].values.astype(float)
        p = pred[col].values.astype(float)
        ok = ~(np.isnan(o) | np.isnan(p))
        o, p = o[ok], p[ok]
        denom = ((o - o.mean()) ** 2).sum()
        nse = float(1 - ((p - o) ** 2).sum() / denom) if denom > 0 else float("nan")
        degenerate = bool(denom <= 1e-9)
        if not degenerate:
            usable.append(col)
        rows[col] = {
            "NSE": nse,
            "RMSE": float(np.sqrt(((p - o) ** 2).mean())),
            "MAE": float(np.abs(p - o).mean()),
            "n": int(ok.sum()),
            "degenerate_obs": degenerate,
            "obs_zero_pct": round(float((o == 0).mean() * 100), 1),
        }
    o_all = np.concatenate([obs[c].values.astype(float) for c in usable]) if usable else np.array([])
    p_all = np.concatenate([pred[c].values.astype(float) for c in usable]) if usable else np.array([])
    ok = ~(np.isnan(o_all) | np.isnan(p_all))
    o_all, p_all = o_all[ok], p_all[ok]
    denom = ((o_all - o_all.mean()) ** 2).sum()
    rows["_pooled"] = {
        "NSE": float(1 - ((p_all - o_all) ** 2).sum() / denom) if denom > 0 else float("nan"),
        "RMSE": float(np.sqrt(((p_all - o_all) ** 2).mean())),
        "MAE": float(np.abs(p_all - o_all).mean()),
        "n": int(ok.sum()),
        "excluded_degenerate": [c for c in obs.columns if c not in usable],
    }
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default_config.yaml"))
    args = parser.parse_args()

    import yaml

    with open(args.config) as f:
        config = yaml.safe_load(f)

    wris_dir = PROJECT_ROOT / config["data"]["wris_data_dir"]
    val_years = config["data"].get("val_years", [2023])
    test_years = config["data"].get("test_years", [2024])
    train_end_year = int(config["data"].get("train_end", "2022-12-31")[:4])

    import yaml as _yaml
    with open(PROJECT_ROOT / config["data"]["reservoirs_file"]) as f:
        reservoir_ids = [r["id"] for r in _yaml.safe_load(f)["reservoirs"]]

    daily = load_daily_inflow(wris_dir, reservoir_ids)
    targets = weekly_sum_next7(daily)
    persist = persistence_forecast(daily)
    clim = climatology_forecast(daily, targets, train_end_year=train_end_year)

    results = {"config": {"wris_dir": str(wris_dir), "val_years": val_years, "test_years": test_years}}
    for name, pred in [("persistence", persist), ("climatology", clim)]:
        results[name] = {}
        for split, years in [("val", val_years), ("test", test_years)]:
            mask = targets.index.year.isin(years)
            obs = targets[mask]
            p = pred[mask].dropna(how="all")
            obs = obs.loc[p.index]
            results[name][split] = metrics_table(obs, p)

    out_dir = PROJECT_ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    (out_dir / "baseline_metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    # Console table (test split)
    print("=" * 100)
    print(f"BASELINES — next-7-day inflow volume (m3/s * days) — test years {test_years}")
    print("=" * 100)
    print(f"{'reservoir':<22}{'PERSIST NSE':>12}{'PERSIST RMSE':>13}{'CLIM NSE':>10}{'CLIM RMSE':>11}{'zero%':>7}{'flag':>8}")
    print("-" * 100)
    test_p = results["persistence"]["test"]
    test_c = results["climatology"]["test"]
    for col in reservoir_ids:
        flag = "DEGEN" if test_p[col]["degenerate_obs"] else ""
        print(
            f"{col:<22}{test_p[col]['NSE']:>12.3f}{test_p[col]['RMSE']:>13.1f}"
            f"{test_c[col]['NSE']:>10.3f}{test_c[col]['RMSE']:>11.1f}{test_p[col]['obs_zero_pct']:>7.1f}{flag:>8}"
        )
    print("-" * 100)
    pooled = test_p["_pooled"]
    pooled_c = test_c["_pooled"]
    print(
        f"{'POOLED':<22}{pooled['NSE']:>12.3f}{pooled['RMSE']:>13.1f}"
        f"{pooled_c['NSE']:>10.3f}{pooled_c['RMSE']:>11.1f}{pooled['n']:>6}"
    )
    if pooled["excluded_degenerate"]:
        print(f"  excluded from pooled (degenerate obs): {', '.join(pooled['excluded_degenerate'])}")
    print("=" * 100)
    print(f"Saved: {out_dir / 'baseline_metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
