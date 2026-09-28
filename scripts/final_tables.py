"""Final consolidated tables, with degenerate-dam handling made explicit."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(r"C:\Users\odeda\Desktop\Projects\PBL")
ART = Path(r"C:\Users\odeda\.openclaw-autoclaw\agents\auto-coder\workspace\.openclaw\tmp\art4")
OUT = ART / "outputs" / "outputs"
sys.path.insert(0, str(REPO / "scripts"))
from run_baselines import load_daily_inflow, weekly_sum_next7, persistence_forecast, climatology_forecast

def pnse(o, p):
    o, p = np.asarray(o, float).ravel(), np.asarray(p, float).ravel()
    m = np.isfinite(o) & np.isfinite(p); o, p = o[m], p[m]
    if o.size == 0: return float("nan")
    d = ((o - o.mean()) ** 2).sum()
    return float(1 - ((p - o) ** 2).sum() / d) if d > 0 else float("nan")

ids = [p.stem for p in sorted((REPO / "data/raw/wris_v2").glob("*.csv"))]
daily = load_daily_inflow(REPO / "data/raw/wris_v2", ids)
tgt = weekly_sum_next7(daily)
per = persistence_forecast(daily)
clim = climatology_forecast(daily, tgt, train_end_year=2022)

print("=" * 104)
print("A. GNN vs ML BASELINES -- pooled week-1 inflow NSE, TEST 2024")
print("=" * 104)
mb = json.load(open(OUT / "ml_baselines.json"))
cands = sorted((ART / "runs" / "runs").glob("seed*/predictions_test.npz")) or \
        list((ART / "runs" / "runs").glob("predictions_test.npz"))
gz = cands[0]
print("   GNN predictions:", gz.parent.name)
z = np.load(gz, allow_pickle=True)
gnn_pool = pnse(z["targets"][:, :, 0], z["preds_median"][:, :, 0])
rows = [("GNN (ours)", round(gnn_pool, 3))]
for m in ["persistence", "climatology", "sarima", "random_forest", "lstm"]:
    p = mb[m]["test"].get("_pooled", {}).get("NSE", float("nan"))
    rows.append((m, round(float(p), 3) if p == p else None))
for name, v in rows:
    print(f"   {name:16s} {v}")
gnn_better = all((v is None) or (gnn_pool > v) for _, v in rows[1:])
print(f"   -> GNN beats every baseline on pooled NSE: {gnn_better}")

print()
print("=" * 104)
print("B. ROLLING-ORIGIN FOLDS by climate phase -- pooled week-1 inflow NSE")
print("=" * 104)
out_rows = []
for fd in sorted((OUT / "rolling_origin").glob("fold_*")):
    yr = int(fd.name.split("_")[1])
    npz = fd / "predictions_test.npz"
    if not npz.exists(): continue
    z = np.load(npz, allow_pickle=True)
    names = [str(r) for r in z["reservoirs"]]
    t, p = z["targets"][:, :, 0], z["preds_median"][:, :, 0]
    # degenerate dams: near-zero observed variance in this fold
    degen = [names[j] for j in range(len(names)) if np.nanvar(t[:, j]) < 1e3]
    keep = [j for j in range(len(names)) if names[j] not in degen]
    jjoni = float(pd.read_csv(REPO / "data/raw/enso/combined_climate_indices.csv",
        parse_dates=["Date"]).set_index("Date").resample("D").ffill()["oni"].pipe(
        lambda s: s[(s.index.year == yr) & (s.index.month.isin([6,7,8,9]))].mean()))
    phase = "El Nino" if jjoni >= 0.5 else ("La Nina" if jjoni <= -0.5 else "neutral")
    m = tgt.index.year == yr
    out_rows.append({
        "fold": yr, "phase": phase, "jjas_oni": round(jjoni, 2),
        "GNN_all": round(pnse(t, p), 3),
        "GNN_clean": round(pnse(t[:, keep], p[:, keep]), 3),
        "persist": round(pnse(tgt[m].values, per[m].values), 3),
        "clim": round(pnse(tgt[m].values, clim[m].values), 3),
        "degenerate_dams": ",".join(degen) or "-",
    })
df = pd.DataFrame(out_rows).sort_values("fold")
print(df.to_string(index=False))
print()
print("   Note: 'GNN_clean' drops dams whose observed variance in that fold is")
print("   near zero (drought years), where NSE is undefined/unstable by construction.")

print()
print("=" * 104)
print("C. LEVEL (week-1 pooled storage NSE)")
print("=" * 104)
rc = pd.read_csv(OUT / "level_rulecurve" / "level_rulecurve_pooled.csv")
print(rc[rc.Week == 1].to_string(index=False))
en = json.load(open(OUT / "el_nino" / "el_nino_summary.json"))
print("   operational level, 2023 JJAS:", en["level_operational_week1"]["val_2023_ElNino|jjas"])
print("   operational level, 2023 all :", en["level_operational_week1"]["val_2023_ElNino|all"])
