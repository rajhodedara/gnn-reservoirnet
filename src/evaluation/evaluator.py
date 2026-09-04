import numpy as np
import pandas as pd
from typing import Dict, List
from .metrics import evaluate_model

class Evaluator:
    """
    Full evaluation pipeline for GNN Reservoir Prediction system.
    """
    def __init__(self, reservoir_names: List[str], basin_mapping: Dict[str, str]):
        """
        Initializes the Evaluator.

        Args:
            reservoir_names (List[str]): List of reservoir names.
            basin_mapping (Dict[str, str]): Mapping from reservoir name to basin name.
        """
        self.reservoir_names = reservoir_names
        self.basin_mapping = basin_mapping

    def evaluate(self, observations: np.ndarray, predictions_median: np.ndarray, predictions_ensemble: np.ndarray, oni_values: np.ndarray) -> Dict[str, pd.DataFrame]:
        """
        Generates evaluation report.

        Args:
            observations (np.ndarray): True values of shape (num_samples, num_reservoirs).
            predictions_median (np.ndarray): Median predictions of shape (num_samples, num_reservoirs).
            predictions_ensemble (np.ndarray): Ensemble predictions of shape (num_samples, num_reservoirs, num_quantiles).
            oni_values (np.ndarray): ONI index values of shape (num_samples,).

        Returns:
            Dict[str, pd.DataFrame]: Evaluation dataframes (per-reservoir, per-basin, enso-vs-neutral).
        """
        per_reservoir_metrics = []
        
        for i, res_name in enumerate(self.reservoir_names):
            obs = observations[:, i]
            pred_med = predictions_median[:, i]
            pred_ens = predictions_ensemble[:, i, :]
            
            metrics = evaluate_model(obs, pred_med, pred_ens)
            metrics['Reservoir'] = res_name
            metrics['Basin'] = self.basin_mapping.get(res_name, 'Unknown')
            per_reservoir_metrics.append(metrics)
            
        df_res = pd.DataFrame(per_reservoir_metrics)
        
        # Per-basin analysis
        df_basin = df_res.groupby('Basin')[['CRPS', 'RMSE', 'NSE', 'KGE']].mean().reset_index()
        
        # El Nino vs Neutral
        el_nino_idx = np.where(oni_values >= 0.5)[0]
        neutral_idx = np.where((oni_values > -0.5) & (oni_values < 0.5))[0]
        
        df_enso = pd.DataFrame()
        if len(el_nino_idx) > 0 and len(neutral_idx) > 0:
            enso_metrics = []
            for i, res_name in enumerate(self.reservoir_names):
                obs = observations[:, i]
                pred_med = predictions_median[:, i]
                pred_ens = predictions_ensemble[:, i, :]
                
                # El Nino
                metrics_en = evaluate_model(obs[el_nino_idx], pred_med[el_nino_idx], pred_ens[el_nino_idx])
                metrics_en['Reservoir'] = res_name
                metrics_en['Condition'] = 'El Nino'
                enso_metrics.append(metrics_en)
                
                # Neutral
                metrics_nu = evaluate_model(obs[neutral_idx], pred_med[neutral_idx], pred_ens[neutral_idx])
                metrics_nu['Reservoir'] = res_name
                metrics_nu['Condition'] = 'Neutral'
                enso_metrics.append(metrics_nu)
                
            df_enso = pd.DataFrame(enso_metrics).groupby('Condition')[['CRPS', 'RMSE', 'NSE', 'KGE']].mean().reset_index()

        return {
            'per_reservoir': df_res,
            'per_basin': df_basin,
            'enso_comparison': df_enso
        }
