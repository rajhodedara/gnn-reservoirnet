#!/usr/bin/env python
"""
Independent QA profiler for data/raw/wris/*.csv.

Profiles all reservoir CSVs regardless of schema (legacy 3-col or new 4-col),
checks continuity, synthetic-inflow contamination, seasonality, storage
plausibility vs gross capacity, and inflow-storage mass-balance coherence.

Usage:
    python scripts/qa_wris_data.py [--out runs/data_qa_report.json]
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WRIS_DIR = PROJECT_ROOT / "data" / "raw" / "wris"

# Gross capacities in TMC (1 MCM = 1/28.3168466 TMC) — from configs/reservoirs.yaml
CAPACITY_TMC = {
    "nagarjuna_sagar": 11560 / 28.3168466,
    "srisailam": 8560 / 28.3168466,
    "almatti": 3440 / 28.3168466,
    "tungabhadra": 3760 / 28.3168466,
    "mettur": 2646 / 28.3168466,
    "krishnaraja_sagara": 1400 / 28.3168466,
    "jayakwadi": 2909 / 28.3168466,
    "ujjani": 3140 / 28.3168466,
    "sardar_sarovar": 9500 / 28.3168466,
    "ukai": 7414 / 28.3168466,
}

WINDOW_START = "2010-01-01"
WINDOW_END = "2024-12-31"


def profile_file(path: Path) -> dict:
    df = pd.read_csv(path)
    cols = list(df.columns)
    schema = "new_4col" if "Inflow (cusecs/cumecs)" in cols else ("legacy_3col" if "inflow" in [c.lower() for c in cols] else "storage_only")

    date_col = cols[0]
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).drop_duplicates(subset=[date_col])
    df = df.set_index(date_col)

    # Resolve inflow/storage columns case-insensitively
    inflow_col = next((c for c in df.columns if "inflow" in c.lower()), None)
    storage_col = next((c for c in df.columns if "storage" in c.lower()), None)

    # Restrict to the common 2010-2024 analysis window for comparability
    w = df.loc[WINDOW_START:WINDOW_END].copy()
    full_idx = pd.date_range(WINDOW_START, WINDOW_END, freq="D")
    missing_days = len(full_idx) - len(w)

    res = {
        "file": str(path.relative_to(PROJECT_ROOT)),
        "schema": schema,
        "columns_raw": cols,
        "rows_total": len(df),
        "date_min_total": str(df.index.min().date()) if len(df) else None,
        "date_max_total": str(df.index.max().date()) if len(df) else None,
        "rows_in_window": len(w),
        "missing_days_in_window": int(missing_days),
    }

    if storage_col is not None:
        s = pd.to_numeric(w[storage_col], errors="coerce")
        cap = CAPACITY_TMC.get(path.stem, float("nan"))
        d = s.diff()
        # Storage artifact metrics: day-over-day jumps beyond 10% of gross capacity
        # cannot be physical inflow (record floods rarely exceed ~5%/day net gain).
        artifact_bound = cap * 0.10 if cap == cap else float("inf")
        artifact_days = int((d.abs() > artifact_bound).sum()) if cap == cap else None
        res["storage"] = {
            "col": storage_col,
            "min": round(float(s.min()), 3) if s.notna().any() else None,
            "mean": round(float(s.mean()), 3) if s.notna().any() else None,
            "max": round(float(s.max()), 3) if s.notna().any() else None,
            "capacity_tmc": round(cap, 1) if cap == cap else None,
            "max_over_capacity_pct": round(float((s.max() / cap - 1) * 100), 1) if s.notna().any() and cap == cap and cap > 0 and s.max() > 0 else None,
            "nan_in_window": int(s.isna().sum()),
            "ds_std_tmc_per_day": round(float(d.std()), 2) if d.notna().sum() > 2 else None,
            "ds_max_abs_tmc": round(float(d.abs().max()), 2) if d.notna().sum() > 2 else None,
            "zero_storage_days": int((s == 0).sum()),
            "artifact_days_gt_10pct_cap": artifact_days,
        }

    if inflow_col is not None:
        i = pd.to_numeric(w[inflow_col], errors="coerce")
        zero_pct = float((i == 0).mean() * 100)
        monsoon = i[i.index.month.isin([6, 7, 8, 9])]
        monsoon_share = float(monsoon.sum() / i.sum() * 100) if i.sum() > 0 else None
        nz = i[i > 0]
        last_nz = str(nz.index.max().date()) if len(nz) else None
        # Degenerate tail: no nonzero inflow in the final 365 days of the window
        tail = i.iloc[-365:]
        degenerate_tail = bool((tail == 0).all()) if len(tail) else False

        # Anti-synthetic check: inflow vs max(ΔS, 0)
        if storage_col is not None:
            s = pd.to_numeric(w[storage_col], errors="coerce")
            ds = s.diff()
            pos_ds = ds.clip(lower=0)
            both = pd.concat([i, pos_ds], axis=1).dropna()
            if len(both) > 100 and both.iloc[:, 0].std() > 0:
                corr = float(both.corr().iloc[0, 1])
                match_pct = float(np.isclose(both.iloc[:, 0], both.iloc[:, 1], atol=1e-2, rtol=1e-2).mean() * 100)
            else:
                corr, match_pct = None, None
            # Mass-balance coherence: does inflow at t precede storage gain t->t+1?
            gain = ds.shift(-1).clip(lower=0)
            pair = pd.concat([i, gain], axis=1).dropna()
            coh = float(pair.corr().iloc[0, 1]) if len(pair) > 100 and pair.iloc[:, 0].std() > 0 else None
        else:
            corr, match_pct, coh = None, None, None

        res["inflow"] = {
            "col": inflow_col,
            "min": round(float(i.min()), 3) if i.notna().any() else None,
            "mean": round(float(i.mean()), 3) if i.notna().any() else None,
            "p99": round(float(i.quantile(0.99)), 3) if i.notna().any() else None,
            "max": round(float(i.max()), 3) if i.notna().any() else None,
            "zero_pct": round(zero_pct, 1),
            "monsoon_jjas_share_pct": round(monsoon_share, 1) if monsoon_share is not None else None,
            "nan_in_window": int(i.isna().sum()),
            "corr_inflow_vs_maxDeltaS": round(corr, 3) if corr is not None else None,
            "match_inflow_eq_maxDeltaS_pct": round(match_pct, 1) if match_pct is not None else None,
            "corr_inflow_vs_nextday_storage_gain": round(coh, 3) if coh is not None else None,
            "last_nonzero_inflow": last_nz,
            "degenerate_tail_365d": degenerate_tail,
        }
    return res


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="runs/data_qa_report.json")
    parser.add_argument("--dir", default=None, help="Directory containing reservoir CSVs (default: data/raw/wris)")
    args = parser.parse_args()

    wris_dir = Path(args.dir) if args.dir else DEFAULT_WRIS_DIR
    if not wris_dir.is_absolute():
        wris_dir = PROJECT_ROOT / wris_dir
    files = sorted(wris_dir.glob("*.csv"))
    if not files:
        print(f"No CSVs found in {wris_dir}")
        return 1

    results = [profile_file(f) for f in files]

    # Console summary
    print(f"WRIS DATA QA — {len(results)} files in {wris_dir} — window {WINDOW_START}..{WINDOW_END}")
    print("=" * 110)
    header = f"{'file':<24}{'schema':<13}{'rows_win':>8}{'miss':>5}{'in_mean':>10}{'in_max':>11}{'zero%':>7}{'JJAS%':>7}{'corr_dS':>9}{'match%':>8}{'dS_std':>9}{'dS_max':>9}{'zeroS':>6}{'artif':>6}"
    print(header)
    print("-" * 140)
    for r in results:
        inf = r.get("inflow", {})
        sto = r.get("storage", {})
        print(
            f"{Path(r['file']).stem:<24}{r['schema']:<13}{r['rows_in_window']:>8}{r['missing_days_in_window']:>5}"
            f"{str(inf.get('mean', '-')):>10}{str(inf.get('max', '-')):>11}{str(inf.get('zero_pct', '-')):>7}"
            f"{str(inf.get('monsoon_jjas_share_pct', '-')):>7}{str(inf.get('corr_inflow_vs_maxDeltaS', '-')):>9}"
            f"{str(inf.get('match_inflow_eq_maxDeltaS_pct', '-')):>8}{str(sto.get('ds_std_tmc_per_day', '-')):>9}"
            f"{str(sto.get('ds_max_abs_tmc', '-')):>9}{str(sto.get('zero_storage_days', '-')):>6}"
            f"{str(sto.get('artifact_days_gt_10pct_cap', '-')):>6}"
        )
    print("=" * 110)

    out_path = PROJECT_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Full report written to: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
