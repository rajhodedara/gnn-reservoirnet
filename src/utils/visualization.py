import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Dict, Tuple

def plot_quantile_forecast(dates: np.ndarray, obs: np.ndarray, q10: np.ndarray, q50: np.ndarray, q90: np.ndarray, title: str = "Reservoir Forecast") -> plt.Figure:
    """
    Plots P10/P50/P90 fan chart for a single reservoir.

    Args:
        dates (np.ndarray): Dates for the x-axis.
        obs (np.ndarray): Observed inflows.
        q10 (np.ndarray): 10th percentile predictions.
        q50 (np.ndarray): Median predictions.
        q90 (np.ndarray): 90th percentile predictions.
        title (str): Plot title.

    Returns:
        plt.Figure: The generated figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(dates, q10, q90, color='lightblue', alpha=0.5, label='P10-P90 Confidence Interval')
    ax.plot(dates, q50, color='blue', label='P50 (Median) Forecast')
    ax.plot(dates, obs, color='black', linestyle='--', label='Observed Inflow')
    
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Inflow")
    ax.legend()
    ax.grid(True)
    return fig

def plot_attention_map(attention_weights: Dict[Tuple[int, int], float], node_positions: Dict[int, Tuple[float, float]], title: str = "Spatial Attention Map") -> plt.Figure:
    """
    Plots a spatial heatmap of GAT attention over a basin map.

    Args:
        attention_weights (Dict[Tuple[int, int], float]): Edge attention weights.
        node_positions (Dict[int, Tuple[float, float]]): (x, y) coordinates for nodes.
        title (str): Plot title.

    Returns:
        plt.Figure: The generated figure.
    """
    import networkx as nx
    fig, ax = plt.subplots(figsize=(8, 8))
    G = nx.DiGraph()
    
    for (src, dst), weight in attention_weights.items():
        G.add_edge(src, dst, weight=weight)
        
    edges, weights = zip(*nx.get_edge_attributes(G, 'weight').items())
    
    nx.draw(G, pos=node_positions, edgelist=edges, edge_color=weights, width=2, edge_cmap=plt.cm.Blues, 
            with_labels=True, node_color='lightgreen', node_size=500, ax=ax)
    
    ax.set_title(title)
    return fig

def plot_attribution_bar(group_importance: Dict[str, float], title: str = "Feature Importance") -> plt.Figure:
    """
    Plots feature attribution bar chart from Integrated Gradients.

    Args:
        group_importance (Dict[str, float]): Importance score per feature group.
        title (str): Plot title.

    Returns:
        plt.Figure: The generated figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    groups = list(group_importance.keys())
    scores = list(group_importance.values())
    
    ax.bar(groups, scores, color='coral')
    ax.set_title(title)
    ax.set_ylabel("Attribution Score")
    ax.set_xlabel("Feature Groups")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig

def plot_evaluation_summary(eval_data: Dict[str, pd.DataFrame]) -> plt.Figure:
    """
    Multi-panel evaluation dashboard.

    Args:
        eval_data (Dict[str, pd.DataFrame]): Data from Evaluator.

    Returns:
        plt.Figure: The generated figure.
    """
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # RMSE per basin
    if 'per_basin' in eval_data and not eval_data['per_basin'].empty:
        df_basin = eval_data['per_basin']
        axs[0, 0].bar(df_basin['Basin'], df_basin['RMSE'], color='skyblue')
        axs[0, 0].set_title('Average RMSE per Basin')
        axs[0, 0].tick_params(axis='x', rotation=45)
    
    # El Nino vs Neutral CRPS
    if 'enso_comparison' in eval_data and not eval_data['enso_comparison'].empty:
        df_enso = eval_data['enso_comparison']
        axs[0, 1].bar(df_enso['Condition'], df_enso['CRPS'], color=['orange', 'gray'])
        axs[0, 1].set_title('CRPS: El Nino vs Neutral')
        
    # KGE distribution
    if 'per_reservoir' in eval_data and not eval_data['per_reservoir'].empty:
        axs[1, 0].hist(eval_data['per_reservoir']['KGE'].dropna(), bins=10, color='purple', alpha=0.7)
        axs[1, 0].set_title('Distribution of KGE Scores')
        
    # Placeholder for extra panel
    axs[1, 1].text(0.5, 0.5, 'Dashboard Summary', ha='center', va='center', fontsize=15)
    axs[1, 1].axis('off')
    
    plt.tight_layout()
    return fig
