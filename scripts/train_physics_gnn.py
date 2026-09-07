#!/usr/bin/env python
"""train_physics_gnn.py — Physics-informed GNN training for reservoir level forecasting.

Architecture: PhysicsConstrainedReservoirGNN
  - Same GAT + TCN + ClimateCrossAttention backbone as ReservoirGNN
  - net_change_head predicts weekly delta-S [TMC] directly
  - Autoregressive mass-balance rollout inside forward()
  - Trained on observed storage levels (not delta-S residuals)

Splits (matching prior work):
  train: 2010-2022
  val:   2023
  test:  2024  (reported, never used for model selection)

Outputs:
  outputs/physics_gnn_results/summary.csv
  outputs/physics_gnn_results/<dam>.csv

Comparison tables vs the delta-S GradientBoosting model are printed at the end.

Usage:
  python scripts/train_physics_gnn.py [--epochs N] [--lr LR] [--no-cuda]
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings
warnings.filterwarnings("ignore")

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import yaml

from src.models.physics_constrained_gnn import PhysicsConstrainedReservoirGNN
from src.training.physics_loss import PhysicsInformedStorageLoss

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MCM_TO_TMC = 1.0 / 28.3168466
CUMEC_WEEK_TO_TMC = 604800.0 / 1e6 / 28.3168466  # weekly vol for 1 cumec, in TMC
LOOKBACK_DAYS = 90
NUM_WEEKS = 12
NUM_QUANTILES = 3
FORECAST_WEEKS_TO_EVAL = [1, 4, 12]

TRAIN_YEARS = range(2010, 2023)
VAL_YEARS = [2023]
TEST_YEARS = [2024]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nse_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Nash-Sutcliffe Efficiency.  Returns NaN if variance is too low."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true, y_pred = y_true[mask], y_pred[mask]
    if len(y_true) < 5:
        return float("nan")
    var = np.var(y_true)
    if var < 1e-6:
        return float("nan")
    denom = np.sum((y_true - y_true.mean()) ** 2)
    num = np.sum((y_true - y_pred) ** 2)
    return float(1.0 - num / denom)


def load_and_build_dataset(res_config, era5: pd.DataFrame, enso: pd.DataFrame):
    """Load all reservoirs and build aligned weekly samples.

    Returns dicts keyed by split ('train', 'val', 'test') containing:
      X  (n_samples, LOOKBACK_DAYS, n_features) — per-sample lookback
      S0 (n_samples,) — storage at forecast origin
      obs_inflow_weekly (n_samples, NUM_WEEKS) — weekly mean inflow [cumecs]
      targets (n_samples, NUM_WEEKS) — observed storage at each forecast week
      dates (n_samples,) — date of forecast origin
    per reservoir (outer list).

    Each sample corresponds to ONE reservoir at ONE date.
    We process reservoirs independently to keep it simple (no GNN batching
    needed across reservoirs — we use a single fully-connected graph of 10
    nodes at inference/eval but independent time samples).
    """
    data_by_dam = {}

    for res in res_config:
        slug = res["id"]
        gross_tmc = res["gross_capacity_mcm"] * MCM_TO_TMC

        df = pd.read_csv(f"data/raw/wris_v2/{slug}.csv", parse_dates=["Date"])
        df = df.set_index("Date").sort_index()
        df = df[~df.index.duplicated(keep="first")]

        S = df["Storage (TMC/MCM)"]
        inflow = df["Inflow (cusecs/cumecs)"].clip(lower=0)

        # ERA5 columns for this dam
        era5_cols = [f"{slug}_runoff", f"{slug}_evap", f"{slug}_soil_moisture", f"{slug}_rainfall"]
        era5_dam = era5[era5_cols].fillna(0)

        # ENSO aligned to daily
        all_dates = pd.date_range("2010-01-01", "2024-12-31", freq="D")
        enso_daily = enso.reindex(all_dates, method="ffill").fillna(0)

        # Merge all daily features
        feat_df = pd.DataFrame(index=all_dates)
        feat_df["storage"] = S.reindex(all_dates)
        feat_df["inflow"] = inflow.reindex(all_dates).ffill().fillna(0)
        feat_df[era5_cols] = era5_dam.reindex(all_dates).fillna(0)
        feat_df[["oni", "soi", "nino34", "iod"]] = enso_daily[["oni", "soi", "nino34", "iod"]]
        feat_df["month_sin"] = np.sin(2 * np.pi * feat_df.index.month / 12)
        feat_df["month_cos"] = np.cos(2 * np.pi * feat_df.index.month / 12)
        feat_df["doy_sin"] = np.sin(2 * np.pi * feat_df.index.dayofyear / 365.25)
        feat_df["doy_cos"] = np.cos(2 * np.pi * feat_df.index.dayofyear / 365.25)
        feat_df["storage_pct"] = (feat_df["storage"].clip(0, gross_tmc) / gross_tmc).fillna(0)
        feat_df = feat_df.fillna(0)

        # Feature columns (excluding raw storage to avoid target leakage in features)
        feature_cols = [c for c in feat_df.columns if c != "storage"]

        # Build samples
        records = {"train": [], "val": [], "test": []}

        dates = feat_df.index
        for i in range(LOOKBACK_DAYS, len(dates) - NUM_WEEKS * 7 - 1):
            t0 = dates[i]
            year = t0.year

            if year in TRAIN_YEARS:
                split = "train"
            elif year in VAL_YEARS:
                split = "val"
            elif year in TEST_YEARS:
                split = "test"
            else:
                continue

            # Lookback window
            lookback_feat = feat_df[feature_cols].iloc[i - LOOKBACK_DAYS: i].values  # (90, F)
            if lookback_feat.shape[0] != LOOKBACK_DAYS:
                continue

            # Current storage S(t0)
            s0 = feat_df["storage"].iloc[i]
            if not np.isfinite(s0):
                continue

            # Observed storage at forecast horizons (end of each week)
            target_indices = [i + w * 7 for w in range(1, NUM_WEEKS + 1)]
            if target_indices[-1] >= len(dates):
                continue
            target_storage = feat_df["storage"].iloc[target_indices].values.copy()
            # Targets clipped to [0, gross_tmc]
            target_storage = np.clip(target_storage, 0, gross_tmc)

            # Weekly mean inflow for each forecast week [cumecs]
            obs_inflow_weekly = np.zeros(NUM_WEEKS)
            for w in range(NUM_WEEKS):
                w_start = i + w * 7
                w_end = i + (w + 1) * 7
                if w_end <= len(dates):
                    obs_inflow_weekly[w] = feat_df["inflow"].iloc[w_start:w_end].mean()

            records[split].append({
                "features": lookback_feat.astype(np.float32),
                "s0": float(s0),
                "inflow": obs_inflow_weekly.astype(np.float32),
                "targets": target_storage.astype(np.float32),
                "date": t0,
            })

        data_by_dam[slug] = {
            "records": records,
            "gross_tmc": gross_tmc,
            "n_features": len(feature_cols),
            "feature_cols": feature_cols,
        }

    return data_by_dam


def records_to_tensors(records):
    """Convert list of record dicts to tensors."""
    if not records:
        return None
    features = torch.tensor(np.stack([r["features"] for r in records]), dtype=torch.float32)
    s0 = torch.tensor([r["s0"] for r in records], dtype=torch.float32)
    inflow = torch.tensor(np.stack([r["inflow"] for r in records]), dtype=torch.float32)
    targets = torch.tensor(np.stack([r["targets"] for r in records]), dtype=torch.float32)
    return {"features": features, "s0": s0, "inflow": inflow, "targets": targets}


def build_single_node_model(n_features, gross_tmc, device):
    """Build a PhysicsConstrainedReservoirGNN treating each reservoir independently.

    For the physics-constrained approach we train one model per dam (or a
    shared model is more natural), but given the small dataset (10 dams) and
    no shared embedding requirement for the standalone baseline, we build a
    simplified architecture where N=1 (single node) and a 2-node fully
    connected graph (self-loop only).

    The model architecture is deliberately kept compact to avoid overfitting
    on the ~13-year training split.
    """
    config = {
        "spatial_in_channels": n_features,
        "spatial_hidden": 32,
        "spatial_out": 32,
        "tcn_in_channels": n_features,
        "tcn_channels": [64, 64],
        "climate_input": LOOKBACK_DAYS,
        "climate_embed": 32,
        "num_weeks": NUM_WEEKS,
        "num_quantiles": NUM_QUANTILES,
        "head_hidden": 64,
        "gat_heads": 2,
        "tcn_kernel": 3,
        "gross_caps_tmc": [gross_tmc],  # 1 node
    }
    model = PhysicsConstrainedReservoirGNN(config)
    return model.to(device)


def train_single_dam(
    slug: str,
    gross_tmc: float,
    records_train,
    records_val,
    n_features: int,
    device: torch.device,
    epochs: int = 80,
    lr: float = 3e-4,
    patience: int = 15,
    batch_size: int = 64,
) -> nn.Module:
    """Train one physics-constrained model per dam."""
    model = build_single_node_model(n_features, gross_tmc, device)
    criterion = PhysicsInformedStorageLoss(alpha=0.85, median_idx=1)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # Single-node edge: just a self-loop so GAT can run
    edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=device)
    # Fake climate indices of shape (B, 4, lookback) — extracted from features
    # We embed 4 climate feature channels specially for climate cross-attention

    train_data = records_to_tensors(records_train)
    val_data = records_to_tensors(records_val)

    if train_data is None or len(records_train) < 10:
        print(f"  [WARN] {slug}: insufficient training data ({len(records_train)} samples)")
        return model

    best_val_loss = float("inf")
    patience_counter = 0
    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    n_train = len(records_train)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train)
        epoch_loss = 0.0
        n_batches = 0

        for batch_start in range(0, n_train, batch_size):
            idx = perm[batch_start: batch_start + batch_size]
            feat_b = train_data["features"][idx].to(device)   # (B, 90, F)
            s0_b = train_data["s0"][idx].to(device)           # (B,)
            inflow_b = train_data["inflow"][idx].to(device)   # (B, W)
            tgt_b = train_data["targets"][idx].to(device)     # (B, W)

            B = feat_b.shape[0]
            # Reshape to (B, N=1, lookback, F)
            node_features = feat_b.unsqueeze(1)  # (B, 1, 90, F)

            # Climate indices: use the 4 ENSO/IOD columns from features
            # Features order: ['inflow', era5_cols(4), 'oni', 'soi', 'nino34', 'iod', 'month_sin', ...]
            # We select the climate sub-slice (columns 5-8: oni/soi/nino34/iod)
            # The climate module expects (B, 4, window=lookback)
            climate_feat_idx = _climate_feat_indices(n_features)
            if climate_feat_idx is not None:
                climate_indices = feat_b[:, :, climate_feat_idx].permute(0, 2, 1)  # (B, 4, 90)
            else:
                climate_indices = torch.zeros(B, 4, LOOKBACK_DAYS, device=device)

            # current_storage: (B, N=1)
            current_storage = s0_b.unsqueeze(1)  # (B, 1)

            optimizer.zero_grad()
            pred_storage = model(node_features, climate_indices, edge_index, current_storage)
            # pred_storage: (B, 1, W, Q)

            # Squeeze node dim for loss
            pred_sq = pred_storage.squeeze(1)   # (B, W, Q)
            tgt_sq = tgt_b                       # (B, W)
            inflow_sq = inflow_b.unsqueeze(0).expand_as(tgt_b.unsqueeze(0)).squeeze(0)  # (B, W) already

            # Loss needs (B, N, ...) — add N=1 dim back
            loss = criterion(
                pred_sq.unsqueeze(1),     # (B, 1, W, Q)
                tgt_sq.unsqueeze(1),      # (B, 1, W)
                current_storage,          # (B, 1)
                inflow_sq.unsqueeze(1),   # (B, 1, W)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        if val_data is not None and len(records_val) > 0:
            val_loss = _evaluate_loss(model, val_data, edge_index, criterion, device, n_features)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stop at epoch {epoch+1}, best_val={best_val_loss:.4f}")
                break

        if (epoch + 1) % 20 == 0:
            avg_train = epoch_loss / max(n_batches, 1)
            val_str = f"{val_loss:.4f}" if val_data is not None and len(records_val) > 0 else "N/A"
            print(f"  Epoch {epoch+1:3d}: train={avg_train:.4f}  val={val_str}")

    # Restore best
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return model


def _climate_feat_indices(n_features: int):
    """Return the indices of [oni, soi, nino34, iod] in the feature vector.
    
    Feature order:
      0: inflow
      1: {slug}_runoff
      2: {slug}_evap
      3: {slug}_soil_moisture
      4: {slug}_rainfall
      5: oni
      6: soi
      7: nino34
      8: iod
      9: month_sin
      10: month_cos
      11: doy_sin
      12: doy_cos
      13: storage_pct
    """
    # indices 5-8 are oni, soi, nino34, iod
    if n_features >= 9:
        return [5, 6, 7, 8]
    return None


def _evaluate_loss(model, data_dict, edge_index, criterion, device, n_features):
    model.eval()
    total = 0.0
    n_batches = 0
    batch_size = 128
    n = data_dict["features"].shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            feat_b = data_dict["features"][start:start+batch_size].to(device)
            s0_b = data_dict["s0"][start:start+batch_size].to(device)
            inflow_b = data_dict["inflow"][start:start+batch_size].to(device)
            tgt_b = data_dict["targets"][start:start+batch_size].to(device)

            B = feat_b.shape[0]
            node_features = feat_b.unsqueeze(1)
            current_storage = s0_b.unsqueeze(1)

            climate_feat_idx = _climate_feat_indices(n_features)
            if climate_feat_idx is not None:
                climate_indices = feat_b[:, :, climate_feat_idx].permute(0, 2, 1)
            else:
                climate_indices = torch.zeros(B, 4, LOOKBACK_DAYS, device=device)

            pred_storage = model(node_features, climate_indices, edge_index, current_storage)
            pred_sq = pred_storage.squeeze(1)
            loss = criterion(
                pred_sq.unsqueeze(1),
                tgt_b.unsqueeze(1),
                current_storage,
                inflow_b.unsqueeze(1),
            )
            total += loss.item()
            n_batches += 1
    return total / max(n_batches, 1)


@torch.no_grad()
def predict_split(model, data_dict, edge_index, device, n_features, gross_tmc):
    """Run inference and return (n_samples, NUM_WEEKS) median-quantile predictions."""
    model.eval()
    all_preds = []
    batch_size = 256
    n = data_dict["features"].shape[0]
    for start in range(0, n, batch_size):
        feat_b = data_dict["features"][start:start+batch_size].to(device)
        s0_b = data_dict["s0"][start:start+batch_size].to(device)

        B = feat_b.shape[0]
        node_features = feat_b.unsqueeze(1)
        current_storage = s0_b.unsqueeze(1)

        climate_feat_idx = _climate_feat_indices(n_features)
        if climate_feat_idx is not None:
            climate_indices = feat_b[:, :, climate_feat_idx].permute(0, 2, 1)
        else:
            climate_indices = torch.zeros(B, 4, LOOKBACK_DAYS, device=device)

        pred_storage = model(node_features, climate_indices, edge_index, current_storage)
        # pred_storage: (B, 1, W, Q) — take median (idx=1)
        pred_median = pred_storage[:, 0, :, 1].cpu().numpy()  # (B, W)
        pred_median = np.clip(pred_median, 0, gross_tmc)
        all_preds.append(pred_median)

    return np.concatenate(all_preds, axis=0)


def compute_baselines(records, gross_tmc, train_records):
    """Compute persistence and climatology baselines for a split."""
    n = len(records)
    persistence = np.zeros((n, NUM_WEEKS))
    climatology = np.zeros((n, NUM_WEEKS))

    # Build climatology from train records
    clim_by_week_doy = {}
    for r in train_records:
        for w_idx, w in enumerate(range(1, NUM_WEEKS + 1)):
            target_date = r["date"] + pd.Timedelta(days=7 * w)
            doy = target_date.dayofyear
            key = (w_idx, doy)
            if key not in clim_by_week_doy:
                clim_by_week_doy[key] = []
            clim_by_week_doy[key].append(r["targets"][w_idx])

    clim_mean = {k: float(np.mean(v)) for k, v in clim_by_week_doy.items()}
    grand_mean = np.mean([r["targets"] for r in train_records]) if train_records else 0.0

    for i, r in enumerate(records):
        for col in range(NUM_WEEKS):
            persistence[i, col] = float(np.clip(r["s0"], 0, gross_tmc))
            target_date = r["date"] + pd.Timedelta(days=7 * (col + 1))
            doy = target_date.dayofyear
            persistence[i, col] = float(np.clip(r["s0"], 0, gross_tmc))
            clim_val = clim_mean.get((col, doy), grand_mean)
            climatology[i, col] = float(np.clip(clim_val, 0, gross_tmc))

    return persistence, climatology


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80, help="Training epochs per dam")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--no-cuda", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.no_cuda or not torch.cuda.is_available() else "cuda")
    print(f"Device: {device}")

    # Load configs
    with open("configs/reservoirs.yaml") as f:
        res_config = yaml.safe_load(f)["reservoirs"]

    enso = pd.read_csv("data/raw/enso/combined_climate_indices.csv")
    enso["Date"] = pd.to_datetime(enso["Date"], errors="coerce")
    enso = enso.dropna(subset=["Date"]).set_index("Date")
    # Fill Year/Month NaN rows from date index
    enso["Year"] = enso.index.year
    enso["Month"] = enso.index.month
    enso = enso.fillna(method="ffill").fillna(0)

    era5 = pd.read_csv("data/raw/era5/reservoir_era5_daily.csv", parse_dates=["Date"])
    era5 = era5.set_index("Date")

    out_dir = "outputs/physics_gnn_results"
    os.makedirs(out_dir, exist_ok=True)

    print("Loading and building datasets...")
    data_by_dam = load_and_build_dataset(res_config, era5, enso)

    # Try to load delta-S model results for comparison
    delta_s_path = "outputs/level_results/summary.csv"
    delta_s_results = None
    if os.path.exists(delta_s_path):
        delta_s_results = pd.read_csv(delta_s_path)
        print(f"Loaded delta-S baseline from {delta_s_path}")
    else:
        print(f"No delta-S baseline found at {delta_s_path} (run scripts/train_level_model.py first)")

    summary_rows = []

    for res in res_config:
        slug = res["id"]
        print(f"\n{'='*60}")
        print(f"Training: {slug}")
        print(f"{'='*60}")

        dam_data = data_by_dam[slug]
        gross_tmc = dam_data["gross_tmc"]
        n_features = dam_data["n_features"]

        records_train = dam_data["records"]["train"]
        records_val = dam_data["records"]["val"]
        records_test = dam_data["records"]["test"]

        print(f"  Samples: train={len(records_train)}, val={len(records_val)}, test={len(records_test)}")

        if len(records_train) < 20:
            print(f"  [SKIP] Insufficient training samples.")
            continue

        model = train_single_dam(
            slug=slug,
            gross_tmc=gross_tmc,
            records_train=records_train,
            records_val=records_val,
            n_features=n_features,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
        )

        # Predict on test split
        if not records_test:
            print(f"  [WARN] No test samples for {slug}")
            continue

        edge_index = torch.tensor([[0], [0]], dtype=torch.long, device=device)

        for split_name, records_split in [("val", records_val), ("test", records_test)]:
            if not records_split:
                continue

            split_data = records_to_tensors(records_split)
            preds = predict_split(model, split_data, edge_index, device, n_features, gross_tmc)
            persistence, climatology = compute_baselines(records_split, gross_tmc, records_train)

            for w_idx, w in enumerate(FORECAST_WEEKS_TO_EVAL):
                w_col = w - 1
                y_pred = preds[:, w_col]
                y_true = np.array([r["targets"][w_col] for r in records_split], dtype=float)
                y_pers = persistence[:, w_col]
                y_clim = climatology[:, w_col]

                # Filter valid targets
                valid = np.isfinite(y_true)
                if valid.sum() < 5:
                    continue

                level_nse = nse_score(y_true[valid], y_pred[valid])
                pers_nse = nse_score(y_true[valid], y_pers[valid])
                clim_nse = nse_score(y_true[valid], y_clim[valid])

                row = {
                    "dam": slug,
                    "split": split_name,
                    "week": w,
                    "level_nse": level_nse,
                    "pers_nse": pers_nse,
                    "clim_nse": clim_nse,
                    "n_samples": int(valid.sum()),
                    "gross_tmc": gross_tmc,
                }
                summary_rows.append(row)

                if split_name == "test":
                    print(f"  Week {w:2d}: level_NSE={level_nse:+.3f}  pers_NSE={pers_nse:+.3f}  clim_NSE={clim_nse:+.3f}  "
                          f"beats_pers={'YES' if level_nse > pers_nse else 'NO '}")

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(f"{out_dir}/summary.csv", index=False)
    print(f"\nSummary written to {out_dir}/summary.csv")

    # Per-dam CSVs
    for res in res_config:
        slug = res["id"]
        dam_df = summary_df[summary_df["dam"] == slug]
        if len(dam_df) > 0:
            dam_df.to_csv(f"{out_dir}/{slug}.csv", index=False)

    # -------------------------------------------------------------------------
    # Print comparison tables
    # -------------------------------------------------------------------------
    print("\n" + "="*70)
    print("PHYSICS GNN RESULTS — TEST SPLIT")
    print("="*70)

    test_df = summary_df[summary_df["split"] == "test"]

    for w in FORECAST_WEEKS_TO_EVAL:
        w_df = test_df[test_df["week"] == w].copy()
        if w_df.empty:
            continue
        print(f"\n--- Week {w} ahead ---")
        print(f"{'Dam':<25} {'PhysGNN':>8} {'Persist':>8} {'Clim':>8} {'BeatPers':>9}")
        beats = 0
        for _, row in w_df.iterrows():
            lnse = f"{row['level_nse']:+.3f}" if np.isfinite(row["level_nse"]) else "  nan"
            pnse = f"{row['pers_nse']:+.3f}" if np.isfinite(row["pers_nse"]) else "  nan"
            cnse = f"{row['clim_nse']:+.3f}" if np.isfinite(row["clim_nse"]) else "  nan"
            bp = row["level_nse"] > row["pers_nse"] if (
                np.isfinite(row["level_nse"]) and np.isfinite(row["pers_nse"])) else False
            beats += int(bp)
            print(f"  {row['dam']:<23} {lnse:>8} {pnse:>8} {cnse:>8} {'YES' if bp else 'NO ':>9}")
        print(f"  Beats persistence: {beats}/{len(w_df)}")

    # Acceptance bar check
    print("\n" + "="*70)
    print("ACCEPTANCE BAR CHECK")
    print("="*70)
    for target_w, min_beats in [(1, 6), (4, 4)]:
        w_df = test_df[test_df["week"] == target_w]
        if w_df.empty:
            print(f"  Week {target_w}: no data")
            continue
        valid_rows = w_df[w_df["level_nse"].notna() & w_df["pers_nse"].notna()]
        beats = (valid_rows["level_nse"] > valid_rows["pers_nse"]).sum()
        total = len(valid_rows)
        status = "PASS" if beats >= min_beats else "FAIL"
        print(f"  Week {target_w}: {beats}/{total} beat persistence (need {min_beats}) — {status}")

    # Comparison with delta-S model
    if delta_s_results is not None:
        print("\n" + "="*70)
        print("COMPARISON: PhysicsGNN vs delta-S GradientBoosting (test split)")
        print("="*70)
        delta_test = delta_s_results[delta_s_results["split"] == "test"]
        for w in FORECAST_WEEKS_TO_EVAL:
            phys_w = test_df[test_df["week"] == w][["dam", "level_nse"]].set_index("dam")
            delta_w = delta_test[delta_test["week"] == w][["dam", "level_nse"]].set_index("dam")
            print(f"\n--- Week {w} ---")
            print(f"{'Dam':<25} {'PhysGNN':>9} {'DeltaS-GB':>10} {'Winner':>7}")
            for dam in phys_w.index:
                p = phys_w.loc[dam, "level_nse"] if dam in phys_w.index else float("nan")
                d = delta_w.loc[dam, "level_nse"] if dam in delta_w.index else float("nan")
                if np.isfinite(p) and np.isfinite(d):
                    winner = "Phys" if p > d else "DeltaS" if d > p else "Tie"
                else:
                    winner = "?"
                ps = f"{p:+.3f}" if np.isfinite(p) else "  nan"
                ds = f"{d:+.3f}" if np.isfinite(d) else "  nan"
                print(f"  {dam:<23} {ps:>9} {ds:>10} {winner:>7}")


if __name__ == "__main__":
    main()
