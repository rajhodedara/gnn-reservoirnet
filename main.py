"""GNN-ReservoirNet: Main Entry Point.

Orchestrates the full pipeline for spatiotemporal prediction of
reservoir levels in Peninsular India during El Niño events.

Usage:
    # Full training
    python main.py --config configs/default_config.yaml

    # Evaluate a trained checkpoint
    python main.py --config configs/default_config.yaml --evaluate --checkpoint runs/best_model.pt

    # Export to ONNX
    python main.py --config configs/default_config.yaml --export-onnx --checkpoint runs/best_model.pt
"""

import argparse
import logging
import random
import sys
import os
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.graph_builder import build_reservoir_graph
from src.data.dataset import ReservoirInflowDataset, create_temporal_splits
from src.data.enso_loader import create_climate_tensor, classify_enso_phase

from src.models.reservoir_gnn import ReservoirGNN
from src.training.trainer import Trainer
from src.training.losses import CombinedLoss
from src.evaluation.evaluator import Evaluator
from src.explainability.integrated_gradients import IGExplainer
from src.explainability.attention_maps import AttentionMapExtractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("GNN-ReservoirNet")


def load_config(config_path: str) -> dict:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    logger.info(f"Loaded configuration from {config_path}")
    
    # Inject reservoirs into config
    if "data" in config and "reservoirs_file" in config["data"]:
        reservoirs_path = config["data"]["reservoirs_file"]
        if os.path.exists(reservoirs_path):
            with open(reservoirs_path, "r") as f:
                res_config = yaml.safe_load(f)
                config["reservoirs"] = res_config.get("reservoirs", [])
                
    return config


def build_graph(config: dict) -> object:
    """Build the reservoir graph from configuration.

    Args:
        config: Full configuration dictionary.

    Returns:
        PyTorch Geometric Data object representing the reservoir graph.
    """
    reservoirs_path = config["data"]["reservoirs_file"]
    with open(reservoirs_path, "r") as f:
        reservoirs_config = yaml.safe_load(f)

    graph = build_reservoir_graph(
        reservoirs=reservoirs_config["reservoirs"],
        correlation_threshold=config["graph"]["correlation_threshold"],
        block_cross_ghats=config["graph"]["block_cross_ghats"],
        use_physical=config["graph"]["physical_edges"],
        use_climatological=config["graph"]["climatological_edges"],
    )
    logger.info(
        f"Built graph: {graph.num_nodes} nodes, "
        f"{graph.edge_index.shape[1]} edges"
    )
    return graph


def create_model(config: dict, graph: object) -> ReservoirGNN:
    """Instantiate the ReservoirGNN model from configuration.

    Args:
        config: Full configuration dictionary.
        graph: PyTorch Geometric graph object.

    Returns:
        Initialized ReservoirGNN model.
    """
    config["model"]["spatial_in_channels"] = 5
    config["model"]["tcn_in_channels"] = 5
    config["model"]["climate_input"] = config["data"]["lookback_days"]
    
    model = ReservoirGNN(config=config["model"])
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created: {num_params:,} trainable parameters")
    return model


def build_datasets(config: dict, graph: object):
    import os
    import pandas as pd
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from src.data.wris_loader import load_reservoir_data, load_climate_indices
    from src.data.dataset import ReservoirInflowDataset, create_temporal_splits

    wris_dir = config["data"]["wris_data_dir"]
    enso_file = os.path.join(config["data"]["enso_data_dir"], "combined_climate_indices.csv")
    
    # Try to load climate indices
    climate_df = load_climate_indices(enso_file)
    if climate_df.empty:
        # Fallback to zeros if user hasn't provided ENSO data yet
        dates = pd.date_range(start="2005-01-01", end="2025-01-01", freq='D')
        climate_df = pd.DataFrame(0.0, index=dates, columns=['ONI', 'DMI', 'PDO'])
        climate_df.index.name = 'Date'
        logger.warning("Climate indices CSV not found. Using zero-filled climate data.")

    inflow_dict = {}
    storage_dict = {}
    features_dict = {}

    reservoirs = config.get("reservoirs", [{"id": f"R{i:02d}"} for i in range(1, 11)])
    num_nodes = graph.num_nodes
    node_feat_dim = 5 # inflow, storage, rainfall(tp), evap(e), soil_moisture(swvl1)

    import xarray as xr
    import glob
    era5_files = glob.glob(os.path.join(config["data"].get("era5_data_dir", "data/raw/era5"), "*.nc"))
    if era5_files:
        era5_ds = xr.open_mfdataset(era5_files, combine='by_coords')
    else:
        era5_ds = None
        logger.warning("No ERA5 NetCDF files found. Climate features will be zeroed.")

    for i, res in enumerate(reservoirs):
        res_id = res["id"]
        # Look for both .csv and _data.csv
        file_path = os.path.join(wris_dir, f"{res_id}.csv")
        if not os.path.exists(file_path):
            file_path = os.path.join(wris_dir, f"{res_id}_data.csv")

        df = load_reservoir_data(file_path, res_id)
        if df.empty:
            # Fail loudly: silent random-fill produced runs indistinguishable
            # from real ones (review P1-5/P1-10).
            raise FileNotFoundError(
                f"No reservoir data found for {res_id} at {file_path}. "
                "Refusing to train on synthetic placeholder data."
            )

        # Merge ERA5 data if available
        if era5_ds is not None and 'latitude' in res and 'longitude' in res:
            try:
                lat, lon = res['latitude'], res['longitude']
                res_ds = era5_ds.sel(latitude=lat, longitude=lon, method="nearest")
                
                # ERA5 time may have hourly/monthly, but we downloaded daily/hourly? Resample to daily if needed.
                # Assuming ERA5 dataset is daily or we resample:
                
                # Extract features: tp (meters -> mm), e (meters -> mm), swvl1 (m3/m3)
                # Select only numeric columns for resampling to avoid aggregation errors on object dtypes
                res_df = res_ds.to_dataframe().select_dtypes(include=[np.number]).resample('D').mean()
                
                if 'tp' in res_df.columns:
                    df['rainfall'] = res_df['tp'] * 1000.0
                if 'e' in res_df.columns:
                    df['evap'] = res_df['e'] * 1000.0  # Usually negative, leaving as is
                if 'swvl1' in res_df.columns:
                    df['soil_moisture'] = res_df['swvl1']
            except Exception as e:
                logger.warning(f"Failed to extract ERA5 data for {res_id}: {e}")

        # Ensure all expected feature columns exist
        for col in ['inflow', 'storage', 'rainfall', 'evap', 'soil_moisture']:
            if col not in df.columns:
                df[col] = 0.0

        inflow_dict[res_id] = df['inflow']
        storage_dict[res_id] = df['storage']
        
        # We need to flatten features for all nodes to concatenate horizontally
        # Order MUST match node_feat_dim = 5
        feats = df[['inflow', 'storage', 'rainfall', 'evap', 'soil_moisture']]
        feats.columns = [f"{res_id}_{c}" for c in feats.columns]
        features_dict[res_id] = feats

    # Combine across reservoirs
    inflow_df = pd.DataFrame(inflow_dict).fillna(0)
    storage_df = pd.DataFrame(storage_dict).fillna(0)
    features_df = pd.concat(features_dict.values(), axis=1)
    # Interpolate missing dates to avoid zero-imputation artifacts
    features_df = features_df.interpolate(method='time').ffill().bfill()

    # Make sure they align
    common_idx = features_df.index.intersection(climate_df.index)
    if len(common_idx) == 0:
         raise ValueError("No overlapping dates between reservoir data and climate data!")

    features_df = features_df.loc[common_idx]
    climate_df = climate_df.loc[common_idx]
    inflow_df = inflow_df.loc[common_idx]
    storage_df = storage_df.loc[common_idx]

    # Standardize data to prevent NaN losses (FP16 overflow and exploding gradients)
    def standardize(df):
        df = df.fillna(0.0)
        return (df - df.mean()) / (df.std().replace(0, 1) + 1e-8)

    # Keep raw (unstandardized) climate for ENSO stratification: thresholds
    # like ONI >= 0.5 are defined in physical units, not z-scores.
    climate_raw = climate_df.copy()

    # Save target normalization constants for un-scaling during evaluation
    inflow_mean = inflow_df.mean().values
    inflow_std = inflow_df.std().replace(0, 1).values + 1e-8
    normalizer = {"mean": inflow_mean, "std": inflow_std}

    features_df = standardize(features_df)
    climate_df = standardize(climate_df)
    inflow_df = standardize(inflow_df)
    storage_df = standardize(storage_df)

    dataset = ReservoirInflowDataset(
        features_df=features_df,
        climate_df=climate_df,
        inflow_df=inflow_df,
        storage_df=storage_df,
        window_size=config["data"]["lookback_days"],
        target_weeks=config["model"]["quantile_head"]["forecast_steps"],
        num_nodes=num_nodes,
        node_feat_dim=node_feat_dim
    )

    # Use specified validation/test years (keys live under `data:` in the config)
    val_years = config["data"].get("val_years", [2023])
    test_years = config["data"].get("test_years", [2024])
    
    # create_temporal_splits expects DatetimeIndex
    dataset_dates = dataset.dates[dataset.valid_indices]
    train_idx, val_idx, test_idx = create_temporal_splits(dataset_dates, val_years, test_years)

    val_sample_dates = dataset_dates[val_idx]
    oni_col = "ONI" if "ONI" in climate_raw.columns else None
    oni_raw_series = (
        climate_raw[oni_col].reindex(val_sample_dates).astype(float)
        if oni_col else pd.Series(dtype=float)
    )
    normalizer["val_sample_dates"] = val_sample_dates
    normalizer["oni_raw"] = oni_raw_series
    
    train_dataset = Subset(dataset, train_idx)
    val_dataset = Subset(dataset, val_idx)
    test_dataset = Subset(dataset, test_idx)
    
    batch_size = config["training"].get("finetune", {}).get("batch_size", 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, normalizer

def train(config: dict, model: ReservoirGNN, graph: object) -> ReservoirGNN:
    """Run the training pipeline."""
    import torch
    from src.training.losses import CombinedLoss
    from src.training.trainer import Trainer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on device: {device}")

    loss_fn = CombinedLoss(
        quantiles=config["model"]["quantile_head"]["quantiles"],
        scaling_factor=config["training"]["loss"]["asymmetric_weight_base"],
        threshold=config["training"]["loss"]["oni_threshold"],
    )

    trainer = Trainer(
        model=model,
        criterion=loss_fn,
        edge_index=graph.edge_index,
        device=device,
        log_dir="runs"
    )

    logger.info("Building datasets from raw WRIS/ENSO sources...")
    train_loader, val_loader, _, _ = build_datasets(config, graph)

    logger.info("=" * 60)
    logger.info("TRAINING on observed WRIS data (2005-Present)")
    logger.info("=" * 60)

    trainer.train(train_loader, val_loader, epochs=config["training"]["finetune"]["epochs"], 
                  lr=config["training"]["finetune"]["lr"], save_dir="runs", 
                  phase="finetune", patience=config["training"]["finetune"]["patience"],
                  weight_decay=config["training"].get("weight_decay", 0.05))

    return model


def evaluate(config: dict, model: ReservoirGNN, graph: object,
             checkpoint_path: str | None = None, split: str = "val") -> dict:
    import numpy as np
    import pandas as pd
    from src.evaluation.evaluator import Evaluator
    from src.evaluation.unscale import unscale_weekly_sum
    import os
    
    assert split in ("val", "test"), f"split must be 'val' or 'test', got {split!r}"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    
    model.to(device)
    model.eval()

    _, val_loader, test_loader, normalizer = build_datasets(config, graph)
    loader = val_loader if split == "val" else test_loader
    
    # Load reservoir config to map names to basins
    reservoirs_path = config["data"]["reservoirs_file"]
    with open(reservoirs_path, "r") as f:
        reservoirs_config = yaml.safe_load(f)
    
    reservoir_names = [r["name"] for r in reservoirs_config["reservoirs"]]
    basin_mapping = {r["name"]: r["basin"] for r in reservoirs_config["reservoirs"]}

    all_targets = []
    all_preds = []
    all_oni = []

    val_sample_dates = normalizer.get("val_sample_dates")
    oni_raw_series = normalizer.get("oni_raw")
    sample_offset = 0

    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            # preds shape: (batch, num_nodes, forecast_steps, num_quantiles)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                preds = model(batch['node_features'], batch['climate_indices'], graph.edge_index.to(device))
            
            # targets shape: (batch, num_nodes, forecast_steps)
            targets = batch['targets']
            oni = batch['oni']
            
            all_targets.append(targets.cpu().numpy())
            all_preds.append(preds.float().cpu().numpy())
            
            # Stratify ENSO on RAW physical ONI, not z-scores
            n_samples = preds.shape[0]
            if val_sample_dates is not None and len(oni_raw_series) > 0:
                dates_b = val_sample_dates[sample_offset:sample_offset + n_samples]
                all_oni.append(np.asarray(oni_raw_series.loc[dates_b].values, dtype=float))
            else:
                all_oni.append(oni.cpu().numpy())
            sample_offset += n_samples
            
    targets_full = np.concatenate(all_targets, axis=0)   # (S, N, 12) z-space weekly sums
    preds_full = np.concatenate(all_preds, axis=0)       # (S, N, 12, Q) z-space
    oni_values = np.concatenate(all_oni, axis=0)

    # Un-scale weekly sums of daily z-scores. The dataset target is
    # sum_i (x_i - mu) / sigma over 7 days, so the exact inverse is
    # sum_z * sigma + 7 * mu  (the old + 1 * mu under-counted 6 mu).
    mean = normalizer['mean']
    std = normalizer['std']
    targets_full = unscale_weekly_sum(targets_full, mean[:, None], std[:, None])
    preds_full = unscale_weekly_sum(preds_full, mean[:, None, None], std[:, None, None])

    # Week-1 arrays drive the existing per-reservoir / per-basin / ENSO reports
    observations = targets_full[:, :, 0]
    predictions_median = preds_full[:, :, 0, 1]  # quantile 1 is the median (0.1, 0.5, 0.9)
    predictions_ensemble = preds_full[:, :, 0, :]

    evaluator = Evaluator(reservoir_names, basin_mapping)
    results = evaluator.evaluate(observations, predictions_median, predictions_ensemble, oni_values)
    
    output_dir = "runs"
    os.makedirs(output_dir, exist_ok=True)
    suffix = "" if split == "val" else f"_{split}"
    results['per_reservoir'].to_csv(os.path.join(output_dir, f"evaluation_metrics_per_reservoir{suffix}.csv"), index=False)
    results['per_basin'].to_csv(os.path.join(output_dir, f"evaluation_metrics_per_basin{suffix}.csv"), index=False)
    
    if not results['enso_comparison'].empty:
        results['enso_comparison'].to_csv(os.path.join(output_dir, f"evaluation_metrics_enso{suffix}.csv"), index=False)

    # All-12-weeks evaluation: per-week, per-reservoir metrics (design horizon)
    per_week = []
    for w in range(targets_full.shape[2]):
        res_w = evaluator.evaluate(
            targets_full[:, :, w],
            preds_full[:, :, w, 1],
            preds_full[:, :, w, :],
            oni_values,
        )
        df_w = res_w['per_reservoir'].copy()
        df_w.insert(0, 'Week', w + 1)
        per_week.append(df_w)
    by_week = pd.concat(per_week, ignore_index=True)
    by_week.to_csv(os.path.join(output_dir, f"evaluation_metrics_by_week{suffix}.csv"), index=False)
    logger.info("Per-week metrics: %d weeks x %d reservoirs", targets_full.shape[2], len(evaluator.reservoir_names))

    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS SAVED TO runs/ (Week 1 Forecast, split=%s)", split)
    logger.info("=" * 60)
    return results

def explain(config: dict, model: ReservoirGNN, graph: object,
            checkpoint_path: str | None = None) -> None:
    import json
    import os
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    _, val_loader, _, _ = build_datasets(config, graph)
    
    # Grab a single batch
    batch = next(iter(val_loader))
    node_features = batch['node_features'].to(device)
    climate_indices = batch['climate_indices'].to(device)
    edge_index = graph.edge_index.to(device)

    # 1. Integrated Gradients
    ig_explainer = IGExplainer(model, edge_index=edge_index, target_idx=0) # Explaining Week 1 prediction
    attributions = ig_explainer.attribute(node_features, climate_indices)
    
    # Feature group mapping for node features (inflow, storage, rainfall, evap, soil_moisture)
    node_groups = {
        "Hydrology": [0, 1],
        "Meteorology": [2, 3, 4]
    }
    
    node_importance = ig_explainer.aggregate_by_group(attributions['node_features'], node_groups)
    
    climate_feature_dim = attributions['climate_indices'].shape[1]
    if climate_feature_dim == 3:
        # Fallback climate indices [ONI, DMI, PDO]
        climate_groups = {
            "ONI": [0],
            "DMI": [1],
            "PDO": [2]
        }
    else:
        # Configured climate indices
        climate_groups = {name.upper(): [i] for i, name in enumerate(config["model"]["climate"]["enso_indices"][:climate_feature_dim])}
        
    climate_importance = ig_explainer.aggregate_by_group(attributions['climate_indices'], climate_groups)

    # 2. Attention Maps
    attn_extractor = AttentionMapExtractor(model)
    attention_map = attn_extractor.extract_attention(node_features, climate_indices, edge_index)
    
    # Format dict for JSON export (tuple keys are not valid JSON)
    attention_map_json = {f"{src}_{dst}": weight for (src, dst), weight in attention_map.items()}

    report = {
        "feature_importance": {
            "node_features": node_importance,
            "climate_features": climate_importance
        },
        "spatial_attention": attention_map_json
    }
    
    output_dir = "runs"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "explainability_report.json"), "w") as f:
        json.dump(report, f, indent=4)
        
    logger.info("=" * 60)
    logger.info("EXPLAINABILITY REPORT SAVED TO runs/explainability_report.json")
    logger.info("=" * 60)
def export_onnx(config: dict, model: ReservoirGNN, graph: object,
                checkpoint_path: str, output_path: str = "runs/model.onnx") -> None:
    """Export trained model to ONNX format.

    Args:
        config: Full configuration dictionary.
        model: ReservoirGNN model instance.
        graph: PyTorch Geometric graph object.
        checkpoint_path: Path to the trained model checkpoint.
        output_path: Output path for the ONNX model.
    """
    device = torch.device("cpu")
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    num_nodes = graph.num_nodes
    lookback = config["data"]["lookback_days"]
    node_feat_dim = 5  # inflow, storage, rainfall(tp), evap(e), soil_moisture(swvl1)
    num_climate_indices = len(config["model"]["climate"]["enso_indices"])
    num_lags = len(config["model"]["climate"]["lags_months"])

    dummy_node_features = torch.randn(1, num_nodes, lookback, node_feat_dim)
    dummy_climate = torch.randn(1, num_climate_indices, num_lags)

    torch.onnx.export(
        model,
        (dummy_node_features, dummy_climate, graph.edge_index),
        output_path,
        input_names=["node_features", "climate_indices", "edge_index"],
        output_names=["quantile_inflows"],
        dynamic_axes={
            "node_features": {0: "batch_size"},
            "climate_indices": {0: "batch_size"},
            "quantile_inflows": {0: "batch_size"},
        },
        opset_version=17,
    )
    logger.info(f"Model exported to ONNX: {output_path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="GNN-ReservoirNet: Spatiotemporal Reservoir Level Prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config", type=str, default="configs/default_config.yaml",
        help="Path to YAML configuration file",
    )

    parser.add_argument(
        "--evaluate", action="store_true",
        help="Run evaluation only (requires --checkpoint)",
    )
    parser.add_argument(
        "--explain", action="store_true",
        help="Generate explainability reports",
    )
    parser.add_argument(
        "--export-onnx", action="store_true",
        help="Export model to ONNX format (requires --checkpoint)",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="runs/",
        help="Directory for outputs (checkpoints, logs, exports)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (python/numpy/torch) for reproducible multi-seed runs",
    )

    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Reproducibility: seed everything before model construction
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    logger.info("Random seed set to %d", args.seed)

    # Build reservoir graph
    graph = build_graph(config)

    # Create model
    model = create_model(config, graph)

    if args.evaluate:
        if not args.checkpoint:
            logger.error("--evaluate requires --checkpoint")
            sys.exit(1)
        evaluate(config, model, graph, args.checkpoint)

    elif args.export_onnx:
        if not args.checkpoint:
            logger.error("--export-onnx requires --checkpoint")
            sys.exit(1)
        export_onnx(config, model, graph, args.checkpoint,
                     output_path=str(Path(args.output_dir) / "model.onnx"))

    elif args.explain:
        explain(config, model, graph, args.checkpoint)

    else:
        # Full training pipeline
        model = train(config, model, graph)
        # Evaluate the BEST validation checkpoint, not the final weights:
        # trainer.train() leaves the model at the last epoch, while the best
        # weights are saved to runs/best_model_finetune.pt on val improvement.
        best_ckpt = os.path.join("runs", "best_model_finetune.pt")
        if os.path.exists(best_ckpt):
            evaluate(config, model, graph, checkpoint_path=best_ckpt, split="val")
            # Held-out test split: the number compared against the baselines.
            evaluate(config, model, graph, checkpoint_path=best_ckpt, split="test")
        else:
            logger.warning("Best checkpoint %s not found; evaluating final weights (not recommended).", best_ckpt)
            evaluate(config, model, graph)
        explain(config, model, graph)

        logger.info("=" * 60)
        logger.info("Pipeline complete. Outputs saved to: %s", args.output_dir)
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
