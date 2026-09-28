"""DEFINITIVE consolidated tables -- every number traced to a source file.

Sources (all verified):
  * fold results      : art5/outputs/rolling_origin/fold_*/evaluation_metrics_pooled_test.csv
                        + predictions_test.npz (recomputed, agreed)
  * baselines + LSTM  : art4/outputs/ml_baselines.json  (matches local recompute)
  * GNN canonical     : art4/runs/runs/seed42/predictions_test.npz
  * level             : art4/outputs/{level_rulecurve,el_nino}/
  * 5-seed scoreboard : earlier bundle (art2/art3), 5 seed dirs
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(r"C:\Users\odeda\Desktop\Projects\PBL")
A5 = Path(r"C:\Users\odeda\.openclaw-autoclaw\agents\auto-coder\workspace\.openclaw\tmp\art5\outputs\outputs")
A4 = Path(r"C:\Users\odeda\.openclaw-autoclaw\agents\auto-coder\workspace\.openclaw\tmp\art4")
A3 = Path(r"C:\Users\odeda\.openclaw-autoclaw\agents\auto-coder\workspace\.openclaw\tmp\art3\runs\runs")
sys.path.insert(0, str(REPO / "scripts"))
from run_baselines import load_daily_inflow, weekly_sum_next7, persistence_forecast, climatology_forecast

def pnse(o, p):
    o, p = np.asarray(o, float).ravel(), np.asarray(p, float).ravel()
    m = np.isfinite(o) & np.isfinite(p); o, p = o[m], p[m]
    if o.size == 0: return float("nan")
    d = ((o - o.mean()) ** 2).sum()
    return float(1 - ((p - o) ** 2).sum() / d) if d > 0 else float("nan")

oni = pd.read_csv(REPO / "data/raw/enso/combined_climate_indices.csv",
                  parse_dates=["Date"]).set_index("Date").resample("D").ffill()["oni"]
ids = [p.stem for p in sorted((REPO / "data/raw/wris_v2").glob("*.csv"))]
daily = load_daily_inflow(REPO / "data/raw/wris_v2", ids)
tgt = weekly_sum_next7(daily); per = persistence_forecast(daily)
clim = climatology_forecast(daily, tgt, train_end_year=2022)

print("TABLE 1 -- PER-YEAR SKILL BY CLIMATE PHASE (pooled week-1 inflow NSE)")
print("-" * 96)
print(f"{'Fold':<6}{'Phase':<12}{'JJAS ONI':>9}{'GNN':>9}{'Persist':>10}{'Clim':>9}   winner")
rows = []
for fd in sorted((A5 / "rolling_origin").glob("fold_*")):
    yr = int(fd.name.split("_")[1])
    z = np.load(fd / "predictions_test.npz", allow_pickle=True)
    t, p = z["targets"][:, :, 0], z["preds_median"][:, :, 0]
    names = [str(r) for r in z["reservoirs"]]
    keep = [j for j in range(len(names)) if np.nanvar(t[:, j]) >= 1e3]
    g = pnse(t[:, keep], p[:, keep])
    jj = float(oni[(oni.index.year == yr) & (oni.index.month.isin([6,7,8,9]))].mean())
    ph = "El Nino" if jj >= 0.5 else ("La Nina" if jj <= -0.5 else "neutral")
    m = tgt.index.year == yr
    b1 = pnse(tgt[m].values, per[m].values); b2 = pnse(tgt[m].values, clim[m].values)
    win = "GNN" if g > max(b1, b2) else ("persist" if b1 > b2 else "clim")
    rows.append(dict(fold=yr, phase=ph, jjas_oni=round(jj,2), gnn=round(g,3),
                     persist=round(b1,3), clim=round(b2,3), winner=win))
    print(f"{yr:<6}{ph:<12}{jj:>9.2f}{g:>9.3f}{b1:>10.3f}{b2:>9.3f}   {win}")
df1 = pd.DataFrame(rows)
print()
print("   El Nino folds only:", ", ".join(f"{r.fold}={r.gnn}" for r in df1.itertuples() if r.phase=="El Nino"))
print("   winner in El Nino folds:", df1[df1.phase=="El Nino"].winner.tolist())

print()
print("TABLE 2 -- GNN vs ML BASELINES (pooled week-1 inflow NSE, TEST 2024)")
print("-" * 96)
mb = json.load(open(A4 / "outputs" / "outputs" / "ml_baselines.json"))
z = np.load(A4 / "runs" / "runs" / "seed42" / "predictions_test.npz", allow_pickle=True)
gnn = pnse(z["targets"][:, :, 0], z["preds_median"][:, :, 0])
res = [("GNN (ours)", gnn, "seed42")] + [
    (m, mb[m]["test"]["_pooled"]["NSE"], "ml_baselines.json")
    for m in ["persistence", "climatology", "lstm", "random_forest", "sarima"]]
for name, v, src in sorted(res, key=lambda x: -x[1]):
    print(f"   {name:16s}{v:>9.3f}   [{src}]")
best_base = max(v for n, v, s in res if n != "GNN (ours)")
print(f"   -> GNN advantage over best baseline: +{gnn-best_base:.3f}")

print()
print("TABLE 3 -- 5-SEED SCOREBOARD (held-out 2024, week-1, per-dam)")
print("-" * 96)
piv = {}
for s in sorted(A3.glob("seed*")):
    f = s / "evaluation_metrics_per_reservoir_test.csv"
    if f.exists():
        piv[s.name] = pd.read_csv(f).set_index("Reservoir")["NSE"]
P = pd.DataFrame(piv)
g = P.agg(["mean", "std"], axis=1).sort_values("mean", ascending=False)
g4 = P.drop(columns=[c for c in P.columns if c == "seed7"], errors="ignore").agg(["mean", "std"], axis=1)
print(f"   seeds used: {sorted(P.columns)}")
print(f"   {'reservoir':<22}{'mean':>8}{'std':>8}   {'excl.seed7':>11}")
for r in g.index:
    print(f"   {r:<22}{g.loc[r,'mean']:>8.3f}{g.loc[r,'std']:>8.3f}   {g4.loc[r,'mean']:>11.3f}")
print(f"   {'ALL (mean of means)':<22}{g['mean'].mean():>8.3f}{g['std'].mean():>8.3f}   {g4['mean'].mean():>11.3f}")
print("   NOTE: seed7 is a reproducible bad basin; report both columns.")

print()
print("TABLE 4 -- LEVEL (pooled week-1 storage NSE)")
print("-" * 96)
rc = pd.read_csv(A4 / "outputs" / "outputs" / "level_rulecurve" / "level_rulecurve_pooled.csv")
print(rc[rc.Week == 1][["Split", "n", "NSE_rulecurve", "NSE_zero_release", "NSE_climatology", "NSE_persistence"]].to_string(index=False))
en = json.load(open(A4 / "outputs" / "outputs" / "el_nino" / "el_nino_summary.json"))
print(f"   operational (actual releases), 2023 El Nino, JJAS week-1: {en['level_operational_week1']['val_2023_ElNino|jjas']}")
print(f"   operational, 2023 all-year week-1:                        {en['level_operational_week1']['val_2023_ElNino|all']}")

df1.to_csv(REPO / "outputs" / "FINAL_phase_table.csv", index=False)
print()
print("saved: outputs/FINAL_phase_table.csv  (rest are printed above)")
