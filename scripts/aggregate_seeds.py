"""Aggregate multi-seed held-out (test) metrics into mean ± std per reservoir.

Reads runs/seed<seed>/evaluation_metrics_per_reservoir_test.csv for each seed
directory and writes outputs/seed_summary.csv, plus a console table that puts
the GNN next to the persistence / climatology baselines when available.

Usage (from the repo root, after a multi-seed training run):
    python scripts/aggregate_seeds.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RUNS = Path("runs")
OUT = Path("outputs")


def main() -> int:
    seed_dirs = sorted([d for d in RUNS.glob("seed*") if d.is_dir()])
    if not seed_dirs:
        print("No runs/seed*/ directories found. Run the notebook's seed loop first.")
        return 1

    frames = []
    for d in seed_dirs:
        f = d / "evaluation_metrics_per_reservoir_test.csv"
        if not f.exists():
            print(f"WARNING: {f} missing — skipping {d.name}")
            continue
        df = pd.read_csv(f).set_index("Reservoir")
        frames.append(df[["NSE", "RMSE", "MAE"]].rename(columns=lambda c: f"{c}_{d.name}"))
        print(f"loaded {d.name}: {len(df)} reservoirs")

    if len(frames) < 2:
        print("Need at least 2 seeds for mean ± std.")
        return 1

    merged = pd.concat(frames, axis=1)
    summary = pd.DataFrame({
        "NSE_mean": merged.filter(like="NSE").mean(axis=1),
        "NSE_std": merged.filter(like="NSE").std(axis=1),
        "RMSE_mean": merged.filter(like="RMSE").mean(axis=1),
        "MAE_mean": merged.filter(like="MAE").mean(axis=1),
        "seeds": merged.filter(like="NSE").notna().sum(axis=1),
    })
    summary = summary.sort_values("NSE_mean", ascending=False)

    # Baseline context, if a baseline_metrics.json is present
    bl_path = OUT / "baseline_metrics.json"
    if bl_path.exists():
        bl = json.loads(bl_path.read_text(encoding="utf-8"))
        slug_by_name = {
            "Nagarjuna Sagar": "nagarjuna_sagar", "Srisailam": "srisailam",
            "Almatti": "almatti", "Tungabhadra": "tungabhadra", "Mettur": "mettur",
            "Krishnaraja Sagara": "krishnaraja_sagara", "Jayakwadi": "jayakwadi",
            "Ujjani": "ujjani", "Sardar Sarovar": "sardar_sarovar", "Ukai": "ukai",
        }
        persist, clim = {}, {}
        for name, slug in slug_by_name.items():
            if name in summary.index:
                persist[name] = bl["persistence"]["test"].get(slug, {}).get("NSE")
                clim[name] = bl["climatology"]["test"].get(slug, {}).get("NSE")
        summary["persistence_NSE"] = pd.Series(persist)
        summary["climatology_NSE"] = pd.Series(clim)

    OUT.mkdir(exist_ok=True)
    summary.to_csv(OUT / "seed_summary.csv")

    print("=" * 84)
    print("3-SEED HELD-OUT (2024) SUMMARY — next-7-day inflow volume")
    print("=" * 84)
    print(f"{'reservoir':<22}{'GNN mean':>9}{'+/-':>5}{'PERSIST':>9}{'CLIM':>8}")
    print("-" * 84)
    for name, r in summary.iterrows():
        p = f"{r['persistence_NSE']:.3f}" if pd.notna(r.get("persistence_NSE")) else "-"
        c = f"{r['climatology_NSE']:.3f}" if pd.notna(r.get("climatology_NSE")) else "-"
        print(f"{name:<22}{r['NSE_mean']:>9.3f}{r['NSE_std']:>5.3f}{p:>9}{c:>8}")
    pooled_mean = summary["NSE_mean"].mean()
    print("-" * 84)
    print(f"{'MEAN':<22}{pooled_mean:>9.3f}")
    print("=" * 84)
    print(f"Saved: {OUT / 'seed_summary.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
