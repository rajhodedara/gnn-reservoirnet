import numpy as np
import properscoring as ps
from typing import Dict

def crps(obs: np.ndarray, pred_dist: np.ndarray) -> float:
    """
    Continuous Ranked Probability Score.

    Args:
        obs (np.ndarray): Observed values.
        pred_dist (np.ndarray): Ensemble predictions or predictive distribution.

    Returns:
        float: Mean CRPS score.
    """
    return float(np.mean(ps.crps_ensemble(obs, pred_dist)))

def rmse(obs: np.ndarray, pred: np.ndarray) -> float:
    """
    Root Mean Square Error.

    Args:
        obs (np.ndarray): Observed values.
        pred (np.ndarray): Predicted values.

    Returns:
        float: RMSE.
    """
    return float(np.sqrt(np.mean((obs - pred) ** 2)))

def nse(obs: np.ndarray, pred: np.ndarray) -> float:
    """
    Nash-Sutcliffe Efficiency.

    Args:
        obs (np.ndarray): Observed values.
        pred (np.ndarray): Predicted values.

    Returns:
        float: NSE value.
    """
    obs_mean = np.mean(obs)
    numerator = np.sum((obs - pred) ** 2)
    denominator = np.sum((obs - obs_mean) ** 2)
    if denominator == 0:
        return float('nan')
    return float(1 - (numerator / denominator))

def kge(obs: np.ndarray, pred: np.ndarray) -> float:
    """
    Kling-Gupta Efficiency.

    Args:
        obs (np.ndarray): Observed values.
        pred (np.ndarray): Predicted values.

    Returns:
        float: KGE value.
    """
    r = np.corrcoef(obs.flatten(), pred.flatten())[0, 1]
    alpha = np.std(pred) / np.std(obs) if np.std(obs) != 0 else float('nan')
    beta = np.mean(pred) / np.mean(obs) if np.mean(obs) != 0 else float('nan')
    
    return float(1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2))

def threshold_crossing_accuracy(obs_storage: np.ndarray, pred_storage: np.ndarray, threshold: float) -> float:
    """
    Binary accuracy for 'will storage drop below X% by date Y?'.

    Args:
        obs_storage (np.ndarray): Observed storage levels.
        pred_storage (np.ndarray): Predicted storage levels.
        threshold (float): Storage threshold percentage.

    Returns:
        float: Accuracy score (0.0 to 1.0).
    """
    obs_crossing = obs_storage < threshold
    pred_crossing = pred_storage < threshold
    return float(np.mean(obs_crossing == pred_crossing))

def evaluate_model(obs: np.ndarray, pred_median: np.ndarray, pred_ensemble: np.ndarray) -> Dict[str, float]:
    """
    Runs all standard metrics on a dataset.

    Args:
        obs (np.ndarray): Observed values.
        pred_median (np.ndarray): Median predictions.
        pred_ensemble (np.ndarray): Predictive distribution.

    Returns:
        Dict[str, float]: Dictionary of metric values.
    """
    return {
        "CRPS": crps(obs, pred_ensemble),
        "RMSE": rmse(obs, pred_median),
        "NSE": nse(obs, pred_median),
        "KGE": kge(obs, pred_median)
    }
