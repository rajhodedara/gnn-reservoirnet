#!/usr/bin/env python
"""Export the repository's model outputs + datasets into a compact JSON bundle
for the ReservoirNet frontend.

Produces a split-file layout (see README) so the web app can lazy-load:

    <out>/meta.json                 units, splits, provenance
    <out>/reservoirs.json           10 nodes + directed topology edges
    <out>/metrics.json              per_reservoir / per_week / per_basin / enso
    <out>/baselines.json            persistence + climatology skill
    <out>/forecast.json             origin date -> dam -> p10/p50/p90/observed
    <out>/climate.json              monthly ONI/SOI/Nino34/IOD (1948..)
    <out>/daily/<id>.json           observed daily inflow + storage (2010-2024)

Usage:
    python scripts/export_app_data.py \
        --runs-dir runs --outputs-dir outputs \
        --data-dir data/raw --out frontend_data
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent

NODES = [
    "nagarjuna_sagar", "srisailam", "almatti", "tungabhadra", "mettur",
    "krishnaraja_sagara", "jayakwadi", "ujjani", "sardar_sarovar", "ukai",
]
MCM_PER_TMC = 28.3168466


def _round(arr, nd=3):
    """Round a nested numeric structure for size, keeping None for NaN."""
    if isinstance(arr, (list, tuple)):
        return [_round(x, nd) for x in arr]
    if isinstance(arr, float) or isinstance(arr, np.floating):
        v = float(arr)
        return None if (v != v) else round(v, nd)   # NaN -> None
    if isinstance(arr, (int, np.integer)):
        return int(arr)
    if isinstance(arr, np.ndarray):
        return _round(arr.tolist(), nd)
    return arr


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_round(obj), separators=(",", ":")), encoding="utf-8")
    return path.stat().st_size


def load_reservoir_meta(cfg_path: Path):
    res = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["reservoirs"]
    nodes, edges = [], []
    for r in res:
        nodes.append({
            "id": r["id"], "name": r["name"], "basin": r["basin"], "river": r["river"],
            "state": r["state"], "lat": r["latitude"], "lon": r["longitude"],
            "gross_capacity_mcm": r["gross_capacity_mcm"],
            "dead_storage_mcm": r["dead_storage_mcm"],
            "catchment_area_km2": r["catchment_area_km2"],
            "elevation_m": r.get("elevation_m"),
            "tribunal": r.get("tribunal"),
        })
        for up in r.get("upstream", []):
            edges.append({"source": up, "target": r["id"], "kind": "physical"})
    return nodes, edges


def _name_to_id(nodes):
    m = {}
    for n in nodes:
        m[n["name"].lower()] = n["id"]
        m[n["id"].lower()] = n["id"]
        m[n["name"].lower().replace(" ", "_")] = n["id"]
    return m


def export_forecast(runs_dir: Path, nodes, out: Path, split="test"):
    npz_p = runs_dir / f"predictions_{split}.npz"
    if not npz_p.exists():
        print(f"  [skip] {npz_p} not found")
        return None
    z = np.load(npz_p, allow_pickle=True)
    dates = [str(d) for d in z["dates"]]
    names = [str(x) for x in z["reservoirs"]]
    n2i = _name_to_id(nodes)
    ids = [n2i.get(x.lower(), x.lower().replace(" ", "_")) for x in names]
    targets = z["targets"]                      # (S, N, 12)
    q = z["preds_all"] if "preds_all" in z.files else None
    med = z["preds_median"] if "preds_median" in z.files else None

    forecast = {}
    for si, d in enumerate(dates):
        per_dam = {}
        for ni, rid in enumerate(ids):
            entry = {"observed": targets[si, ni].tolist()}
            if q is not None:
                entry["p10"] = q[si, ni, :, 0].tolist()
                entry["p50"] = q[si, ni, :, 1].tolist()
                entry["p90"] = q[si, ni, :, 2].tolist()
            elif med is not None:
                entry["p50"] = med[si, ni].tolist()
            per_dam[rid] = entry
        forecast[d] = per_dam
    size = write_json(out / "forecast.json", forecast)
    print(f"  forecast.json: {len(dates)} origins x {len(ids)} dams  ({size/1e6:.1f} MB)"
          f"{'' if q is not None else '  [median only -- preds_all absent]'}")
    return {"origins": len(dates), "dams": ids, "has_quantiles": q is not None}


def export_metrics(runs_dir: Path, out: Path):
    seed_dirs = sorted(p for p in runs_dir.glob("seed*") if p.is_dir())
    if not seed_dirs:
        print("  [skip] no seed dirs under runs/")
        return
    kinds = {
        "per_reservoir": "evaluation_metrics_per_reservoir_test.csv",
        "per_basin": "evaluation_metrics_per_basin_test.csv",
        "enso": "evaluation_metrics_enso_test.csv",
        "per_week": "evaluation_metrics_by_week_test.csv",
    }
    metrics = {"seeds": [p.name for p in seed_dirs]}
    for kind, fname in kinds.items():
        frames = []
        for sd in seed_dirs:
            p = sd / fname
            if p.exists():
                df = pd.read_csv(p)
                df["seed"] = sd.name
                frames.append(df)
        if not frames:
            continue
        all_df = pd.concat(frames, ignore_index=True)
        key = "Week" if kind == "per_week" else ("Condition" if kind == "enso" else
                                                 ("Basin" if kind == "per_basin" else "Reservoir"))
        num = [c for c in all_df.columns if all_df[c].dtype.kind in "fi" and c != "Week"]
        agg = all_df.groupby(key)[num].agg(["mean", "std"])
        agg.columns = [f"{a}_{b}" for a, b in agg.columns]
        metrics[kind] = agg.reset_index().to_dict(orient="records")
        metrics[kind + "_per_seed"] = all_df.to_dict(orient="records")
    size = write_json(out / "metrics.json", metrics)
    print(f"  metrics.json ({size/1e3:.0f} KB) from {len(seed_dirs)} seeds")


def export_baselines(outputs_dir: Path, out: Path):
    p = outputs_dir / "baseline_metrics.json"
    if not p.exists():
        print(f"  [skip] {p} not found")
        return
    d = json.loads(p.read_text(encoding="utf-8"))
    write_json(out / "baselines.json", d)
    print("  baselines.json")


def export_levels(outputs_dir: Path, out: Path):
    levels = {}
    for name, sub in [("operational", "level_results"), ("physics", "physics_gnn_results")]:
        p = outputs_dir / sub / "summary.csv"
        if p.exists():
            levels[name] = pd.read_csv(p).to_dict(orient="records")
    ro = outputs_dir / "rolling_origin" / "rolling_origin_summary.csv"
    if ro.exists():
        levels["rolling_origin"] = pd.read_csv(ro).to_dict(orient="records")
    if levels:
        write_json(out / "levels.json", levels)
        print(f"  levels.json ({', '.join(levels)})")


def export_climate(data_dir: Path, out: Path):
    p = data_dir / "enso" / "combined_climate_indices.csv"
    if not p.exists():
        print(f"  [skip] {p} not found")
        return
    df = pd.read_csv(p)
    cols = [c for c in ["oni", "soi", "nino34", "iod"] if c in df.columns]
    obj = {"dates": df["Date"].astype(str).tolist(),
           **{c: df[c].tolist() for c in cols}}
    size = write_json(out / "climate.json", obj)
    print(f"  climate.json {len(df)} rows, cols {cols} ({size/1e3:.0f} KB)")


def export_daily(data_dir: Path, nodes, out: Path):
    total = 0
    for n in nodes:
        p = data_dir / "wris_v2" / f"{n['id']}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        obj = {"dates": df["Date"].astype(str).tolist(),
               "inflow": df["Inflow (cusecs/cumecs)"].tolist(),
               "storage_tmc": df["Storage (TMC/MCM)"].tolist()}
        total += write_json(out / "daily" / f"{n['id']}.json", obj)
    print(f"  daily/*.json ({total/1e6:.1f} MB across {len(nodes)} dams)")


def export_era5(data_dir: Path, nodes, out: Path):
    p = data_dir / "era5" / "reservoir_era5_daily.csv"
    if not p.exists():
        print("  [skip] era5 not found")
        return
    df = pd.read_csv(p, parse_dates=["Date"])
    obj = {"dates": df["Date"].dt.strftime("%Y-%m-%d").tolist()}
    for n in nodes:
        for feat in ["rainfall", "runoff", "evap", "soil_moisture"]:
            col = f"{n['id']}_{feat}"
            if col in df.columns:
                obj.setdefault("by_dam", {}).setdefault(n["id"], {})[feat] = df[col].tolist()
    size = write_json(out / "era5.json", obj)
    print(f"  era5.json ({size/1e6:.1f} MB)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--outputs-dir", default="outputs")
    ap.add_argument("--data-dir", default="data/raw")
    ap.add_argument("--out", default="frontend_data")
    ap.add_argument("--config", default="configs/reservoirs.yaml")
    args = ap.parse_args()

    runs, outs = Path(args.runs_dir), Path(args.outputs_dir)
    data, out = Path(args.data_dir), Path(args.out)
    nodes, edges = load_reservoir_meta(PROJECT_ROOT / args.config)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Exporting frontend bundle -> {out}")
    write_json(out / "reservoirs.json", {"reservoirs": nodes, "edges": edges})
    print(f"  reservoirs.json ({len(nodes)} nodes, {len(edges)} physical edges)")

    fc = export_forecast(runs, nodes, out)
    meta = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "units": {"inflow": "m3/s per day (weekly sum)", "storage": "TMC",
                  "mcm_per_tmc": MCM_PER_TMC},
        "splits": {"train_end": "2022-12-31", "val_year": 2023, "test_year": 2024},
        "forecast": fc,
    }
    write_json(out / "meta.json", meta)
    print("  meta.json")

    export_metrics(runs, out)
    export_baselines(outs, out)
    export_levels(outs, out)
    export_climate(data, out)
    export_era5(data, nodes, out)
    export_daily(data, nodes, out)

    total = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
    print(f"Done. {total/1e6:.1f} MB total in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
