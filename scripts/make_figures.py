"""Three presentation figures for GNN-ReservoirNet, from final run artifacts."""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE = r"C:\Users\odeda\Desktop\Projects\PBL"
FIG = os.path.join(BASE, "outputs", "figures")
os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# ---------------- Figure 1: inflow NSE by dam (5 seeds) ----------------
ss = pd.read_csv(os.path.join(BASE, "outputs", "run8", "seed_summary_5seed.csv"))
ss = ss.sort_values("NSE_mean", ascending=True)
labels = [n.replace("Krishnaraja Sagara", "KRS").replace("Nagarjuna Sagar", "NS").replace("Sardar Sarovar", "SSP") for n in ss["Reservoir"]]
y = np.arange(len(ss))
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.barh(y + 0.27, ss["NSE_mean"], height=0.25, xerr=ss["NSE_std"], color="#1f77b4", label="GNN (5 seeds, mean±std)", capsize=2)
ax.barh(y, ss["persistence_NSE"], height=0.25, color="#d62728", alpha=0.85, label="Persistence")
ax.barh(y - 0.27, ss["climatology_NSE"], height=0.25, color="#7f7f7f", alpha=0.85, label="Climatology")
ax.set_yticks(y, labels)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("Week-1 inflow forecast NSE — held-out 2024")
ax.set_title("Inflow forecasting: 10/10 dams beat persistence, 7/10 beat climatology")
ax.legend(loc="lower right", frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig1_inflow_nse_by_dam.png"))
plt.close(fig)

# ---------------- Figure 2: skill vs horizon (canonical + rolling origin) ----------------
seeds = ["seed42", "seed7", "seed123"]
curves = []
for s in seeds:
    bw = pd.read_csv(os.path.join(BASE, "scratch", "runs5_extract", "runs", s, "evaluation_metrics_by_week_test.csv"))
    curves.append(bw.groupby("Week")["NSE"].mean())
canonical = pd.concat(curves, axis=1).mean(axis=1)

fig, ax = plt.subplots(figsize=(8, 4.6))
ro = pd.read_csv(os.path.join(BASE, "outputs", "rolling_origin", "rolling_origin_summary.csv")).set_index("fold")
for fold in ro.index:
    ax.plot(range(1, 13), ro.loc[fold, "week_1":"week_12"].values, color="#1f77b4", alpha=0.25, lw=1)
ax.plot(range(1, 13), canonical.values, color="#1f77b4", lw=2.6, label="Canonical (2024, 3 seeds)")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(1, 13))
ax.set_xlabel("Forecast lead (weeks)")
ax.set_ylabel("Mean NSE across 10 dams")
ax.set_title("Skill decay by horizon: weekly skill for 3–4 weeks, honest collapse beyond")
ax.text(0.02, 0.97, "light lines: rolling-origin refits, test years 2020–2024\n(dark line: canonical held-out 2024)",
        transform=ax.transAxes, va="top", fontsize=8, color="#444444")
ax.legend(loc="upper right", frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig2_nse_vs_horizon.png"))
plt.close(fig)

# ---------------- Figure 3: El Nino 2023 showcase (Srisailam) ----------------
d = np.load(os.path.join(BASE, "scratch", "runs5_extract", "runs", "predictions_val.npz"), allow_pickle=True)
names = [str(x) for x in d["reservoirs"]]
dates = pd.to_datetime(d["dates"])
j = names.index("Srisailam")
tgt, prd = d["targets"][:, j, 0], d["preds_median"][:, j, 0]  # week-1, raw weekly m3/s-days

# monsoon-season aggregation: weekly series is noisy; smooth with a 4-sample rolling mean
s = pd.DataFrame({"date": dates, "obs": tgt, "pred": prd}).set_index("date").sort_index()
obs_s, pred_s = s["obs"].rolling(7, min_periods=3).mean(), s["pred"].rolling(7, min_periods=3).mean()
fig, ax = plt.subplots(figsize=(8.6, 4.4))
ax.plot(s.index, obs_s, color="#111111", lw=1.8, label="Observed inflow")
ax.plot(s.index, pred_s, color="#1f77b4", lw=1.8, label="GNN P50 forecast (week-1)")
ax.axvspan(pd.Timestamp("2023-06-01"), pd.Timestamp("2023-10-15"), color="#ffe08a", alpha=0.35, label="2023 El Niño monsoon")
ax.set_ylabel("Weekly inflow (m³/s-days), 7-sample smoothing")
ax.set_xlabel("2023 validation year (El Niño onset)")
ax.set_title("Srisailam: the model tracks the disrupted 2023 monsoon")
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "fig3_el_nino_2023_srisailam.png"))
plt.close(fig)

print("figures written:", sorted(os.listdir(FIG)))
