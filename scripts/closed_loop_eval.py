"""Closed-loop (state-feedback) storage NSE from run #8c predictions,
using the pipeline's own normalizer for units (inflow_raw + storage_raw).

S_pred(w) = S_obs(t + 7*(w-1)) + I_pred(w) - R_obs(w)   (restart each week)
R_obs(w)  = S_obs(t + 7*(w-1)) + I_obs(w) - S_obs(t + 7*w)
Persistence: S_pred = S_obs(t + 7*(w-1))
"""
import os
import sys

import numpy as np
import pandas as pd

BASE = r"C:\Users\odeda\Desktop\Projects\PBL"
sys.path.insert(0, BASE)
os.chdir(BASE)

import torch  # noqa: E402

import main as M  # noqa: E402

torch.set_num_threads(max(1, os.cpu_count() - 2))

config = M.load_config(os.path.join(BASE, "configs", "default_config.yaml"))
graph = M.build_graph(config) if hasattr(M, "build_graph") else None
out = M.build_datasets(config, graph)
normalizer = out[3]  # (train_loader, val_loader, test_loader, normalizer)

storage_raw = normalizer["storage_raw"]
inflow_raw = normalizer.get("inflow_raw")
test_dates = normalizer["test_sample_dates"]

d = np.load(BASE + r"\scratch\runs5_extract\runs\predictions_test.npz", allow_pickle=True)
targets = d["targets"]      # (S, N, 12) weekly, pipeline raw units
preds = d["preds_median"]
dates = pd.to_datetime(d["dates"])
names = [str(x) for x in d["reservoirs"]]

print("sample dates:", dates[0].date(), "->", dates[-1].date(), f"({len(dates)} samples)")
print("storage_raw units sample (Srisailam tail):")
print(storage_raw["srisailam"].tail(3))

rows = []
n_samples, n_dams, n_weeks = preds.shape
for j, name in enumerate(names):
    slug = name.lower().replace(" ", "_")
    s_ser = storage_raw[slug]
    i_ser = inflow_raw[slug] if inflow_raw and slug in inflow_raw else None
    for si in range(n_samples):
        t = test_dates[si]
        t = pd.Timestamp(t)
        for w in range(1, n_weeks + 1):
            t_prev = t + pd.Timedelta(days=7 * (w - 1))
            t_now = t + pd.Timedelta(days=7 * w)
            if t_now > s_ser.index.max():
                continue
            s_prev = float(s_ser.asof(t_prev))
            s_now = float(s_ser.asof(t_now))
            if not (np.isfinite(s_prev) and np.isfinite(s_now)):
                continue
            # observed weekly inflow in pipeline units, from the targets themselves
            i_obs = float(targets[si, j, w - 1])
            i_pred = float(preds[si, j, w - 1])
            r_obs = s_prev + i_obs - s_now  # releases implied by observed data
            cap = float(s_ser.max())
            s_gnn = float(np.clip(s_prev + i_pred - r_obs, 0.0, cap))
            s_pers = float(np.clip(s_prev, 0.0, cap))
            rows.append(
                {
                    "Reservoir": name,
                    "Week": w,
                    "s_act": s_now,
                    "s_gnn": s_gnn,
                    "s_pers": s_pers,
                }
            )

df = pd.DataFrame(rows)


def nse(g, col):
    act = g["s_act"]
    return 1 - ((g[col] - act) ** 2).sum() / max(((act - act.mean()) ** 2).sum(), 1e-9)


for w in [1, 4, 12]:
    sub = df[df["Week"] == w]
    tbl = sub.groupby("Reservoir").apply(
        lambda g: pd.Series(
            {"GNN_closed": nse(g, "s_gnn"), "Persistence": nse(g, "s_pers"), "n": len(g)}
        ),
        include_groups=False,
    )
    print(f"\n=== WEEK {w} closed-loop storage NSE, test 2024 (pipeline units) ===")
    print(tbl.round(3).to_string())
    print(f"mean: GNN {tbl['GNN_closed'].mean():.3f} vs persistence {tbl['Persistence'].mean():.3f}")
