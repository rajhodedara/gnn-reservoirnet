import json
from pathlib import Path

import pandas as pd

root = Path(r"C:\Users\odeda\Desktop\Projects\PBL")
r7_out = root / "scratch/run7_extract_out/outputs"
r7_runs = root / "scratch/run7_extract_runs/runs"
r6_out = root / "outputs/run6"

print("=== RUN #7 seed_summary (3 seeds, true rainfall tp, Mettur+SSP fixed) ===")
ss7 = pd.read_csv(r7_out / "seed_summary.csv")
print(ss7.round(3).to_string(index=False))
print(f"\nmean all-10: {ss7['NSE_mean'].mean():.3f} | ex-Mettur: {ss7[ss7['Reservoir']!='Mettur']['NSE_mean'].mean():.3f}")

print()
print("=== TRUE-RAINFALL ABLATION: run #7 (tp) vs run #6 (runoff proxy), held-out 2024 ===")
r7 = pd.read_csv(r7_runs / "seed42" / "evaluation_metrics_per_reservoir_test.csv").set_index("Reservoir")["NSE"]
r6s = pd.read_csv(r6_out / "seed_summary.csv").set_index("Reservoir")
bl = json.load(open(r7_out / "baseline_metrics.json"))
persist = {k: v["NSE"] for k, v in bl["persistence"]["test"].items() if not k.startswith("_")}
clim = {k: v["NSE"] for k, v in bl["climatology"]["test"].items() if not k.startswith("_")}
rows = []
for res in r7.index:
    slug = res.lower().replace(" ", "_")
    rows.append((res, r7[res], r6s.loc[res, "NSE_mean"], r6s.loc[res, "NSE_std"], persist.get(slug, float("nan")), clim.get(slug, float("nan"))))
print(f"{'reservoir':<20}{'run7(tp)':>10}{'run6(proxy)':>12}{'delta':>8}{'PERSIST':>9}{'CLIM':>7}")
for res, a, b, sd, p, c in sorted(rows, key=lambda r: -r[1]):
    print(f"{res:<20}{a:>10.3f}{b:>12.3f}{a-b:>+8.3f}{p:>9.3f}{c:>7.3f}")
m7 = sum(r[1] for r in rows) / 10
m6 = sum(r[2] for r in rows) / 10
print(f"{'MEAN':<20}{m7:>10.3f}{m6:>12.3f}{m7-m6:>+8.3f}")
print(f"beats persistence: {sum(1 for r in rows if r[1] > r[4])}/10 | beats climatology: {sum(1 for r in rows if r[1] > r[5])}/10")

print()
print("=== BLENDING (Kaggle-run, held-out 2024) ===")
be = pd.read_csv(r7_out / "blend_eval_full.csv")
g = be.groupby("Week")[["NSE_gnn", "NSE_clim", "NSE_blended"]].mean()
print(g.round(3).to_string())
print(f"\nall-weeks means: GNN={be['NSE_gnn'].mean():.3f}  CLIM={be['NSE_clim'].mean():.3f}  BLENDED={be['NSE_blended'].mean():.3f}")
pos_blended = (g["NSE_blended"] > 0).sum()
print(f"weeks with blended NSE > 0: {pos_blended}/12")

print()
print("=== 12-week GNN-only curve (run #7, true rainfall) ===")
bw = pd.read_csv(r7_runs / "seed42" / "evaluation_metrics_by_week_test.csv")
curve = bw.groupby("Week")["NSE"].mean()
for w, v in curve.items():
    print(f"  week {w:>2}: {v:>7.3f}")
