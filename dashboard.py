import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import yaml
import torch
import networkx as nx
import os

# Set page configuration
st.set_page_config(page_title="GNN-ReservoirNet Dashboard", layout="wide", page_icon="🌊")

st.title("GNN-ReservoirNet: Inflow Forecasting Dashboard")
st.markdown("Interactive visualization for model evaluation, forecasting, and explainability.")

# Define paths
CONFIG_PATH = "configs/default_config.yaml"
EVAL_ENSO_PATH = "runs/evaluation_metrics_enso.csv"
EVAL_RES_PATH = "runs/evaluation_metrics_per_reservoir.csv"
XAI_REPORT_PATH = "runs/explainability_report.json"
CHECKPOINT_PATH = "runs/best_model_finetune.pt"

@st.cache_data
def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

@st.cache_data
def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data
def load_json(path):
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return None

config = load_config()

tab1, tab2, tab3 = st.tabs(["📊 Model Evaluation", "🔮 Forecast Explorer", "🧠 Explainability (XAI)"])

# ==========================================
# TAB 1: MODEL EVALUATION
# ==========================================
with tab1:
    st.header("Offline Evaluation Metrics")
    
    col1, col2 = st.columns(2)
    
    # ENSO Metrics
    with col1:
        st.subheader("Performance by ENSO Condition")
        enso_df = load_csv(EVAL_ENSO_PATH)
        if enso_df is not None:
            fig_enso = px.bar(
                enso_df, 
                x="Condition", 
                y=["CRPS", "RMSE"], 
                barmode="group",
                title="CRPS & RMSE Across El Niño vs Neutral"
            )
            st.plotly_chart(fig_enso, use_container_width=True)
            st.dataframe(enso_df)
        else:
            st.warning("ENSO metrics not found. Run evaluation first.")
            
    # Reservoir Metrics
    with col2:
        st.subheader("Performance per Reservoir")
        res_df = load_csv(EVAL_RES_PATH)
        if res_df is not None:
            # Sort by CRPS
            res_df = res_df.sort_values(by="CRPS")
            fig_res = px.bar(
                res_df, 
                x="Reservoir", 
                y="CRPS", 
                color="Basin",
                title="Continuous Ranked Probability Score (CRPS) per Reservoir"
            )
            st.plotly_chart(fig_res, use_container_width=True)
            st.dataframe(res_df)
        else:
            st.warning("Reservoir metrics not found. Run evaluation first.")


# ==========================================
# TAB 2: FORECAST EXPLORER
# ==========================================
with tab2:
    st.header("Forecast Explorer")
    st.markdown("Pick a batch from the validation set and visualize the multi-week inflow forecasts.")
    
    @st.cache_resource
    def get_model_and_data():
        from main import build_datasets, build_reservoir_graph
        from src.models.reservoir_gnn import ReservoirGNN
        
        # We must set device to CPU for Streamlit to prevent GPU OOM on multiple refreshes
        device = torch.device("cpu")
        
        with open(config["data"]["reservoirs_file"], "r") as f:
            config["reservoirs"] = yaml.safe_load(f)
            
        # Graph
        graph = build_reservoir_graph(config["reservoirs"], config["graph"])
        
        # Data
        _, val_loader, normalizer = build_datasets(config, graph)
        
        # Model
        # Dynamically set config values like in main.py
        config["model"]["tcn_in_channels"] = 6
        config["model"]["climate_input"] = val_loader.dataset.climate.shape[1]
        
        model = ReservoirGNN(config=config["model"])
        if os.path.exists(CHECKPOINT_PATH):
            state_dict = torch.load(CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(state_dict)
            
        model.to(device)
        model.eval()
        
        return model, val_loader, graph, normalizer, device
        
    try:
        model, val_loader, graph, normalizer, device = get_model_and_data()
        
        reservoirs = config.get("reservoirs", [])
        reservoir_names = [res["id"] for res in reservoirs]
        
        col_res, col_batch = st.columns(2)
        with col_res:
            selected_res_name = st.selectbox("Select Reservoir", reservoir_names)
            res_idx = reservoir_names.index(selected_res_name)
            
        with col_batch:
            # For simplicity, we just extract one batch 
            # In a real scenario, we'd allow selecting specific dates
            batch_num = st.number_input("Validation Batch Number", min_value=0, max_value=len(val_loader)-1, value=0)
            
        if st.button("Run Inference"):
            with st.spinner("Running forward pass..."):
                # Get the specific batch
                for i, batch in enumerate(val_loader):
                    if i == batch_num:
                        break
                        
                node_features = batch['node_features'].to(device)
                climate_indices = batch['climate_indices'].to(device)
                targets = batch['targets'].to(device)
                edge_index = graph.edge_index.to(device)
                
                with torch.no_grad():
                    predictions = model(node_features, climate_indices, edge_index)
                
                # Take the first sample in the batch
                sample_idx = 0
                
                # Predictions shape: (batch, nodes, weeks, quantiles)
                # Targets shape: (batch, nodes, weeks)
                pred_sample = predictions[sample_idx, res_idx].cpu().numpy() # (weeks, 3)
                target_sample = targets[sample_idx, res_idx].cpu().numpy()   # (weeks,)
                
                # Unscale predictions if normalizer is available
                if normalizer is not None and "inflow_mean" in normalizer:
                    mean = normalizer["inflow_mean"].cpu().numpy()[res_idx]
                    std = normalizer["inflow_std"].cpu().numpy()[res_idx]
                    
                    pred_sample = (pred_sample * std) + mean
                    target_sample = (target_sample * std) + mean
                    
                    # ReLU to prevent negative inflows
                    pred_sample = np.maximum(pred_sample, 0.0)
                    target_sample = np.maximum(target_sample, 0.0)
                
                weeks = np.arange(1, len(target_sample) + 1)
                
                fig_forecast = go.Figure()
                
                # Add P90 and P10 (uncertainty band)
                fig_forecast.add_trace(go.Scatter(
                    x=np.concatenate([weeks, weeks[::-1]]),
                    y=np.concatenate([pred_sample[:, 2], pred_sample[:, 0][::-1]]),
                    fill='toself',
                    fillcolor='rgba(0,176,246,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    hoverinfo="skip",
                    showlegend=True,
                    name='Uncertainty (P10 - P90)'
                ))
                
                # Add P50 (Median)
                fig_forecast.add_trace(go.Scatter(
                    x=weeks, y=pred_sample[:, 1],
                    mode='lines+markers',
                    line=dict(color='rgb(0,176,246)', width=3),
                    name='Predicted (Median)'
                ))
                
                # Add Ground Truth
                fig_forecast.add_trace(go.Scatter(
                    x=weeks, y=target_sample,
                    mode='lines+markers',
                    line=dict(color='rgb(231,107,243)', width=3, dash='dash'),
                    name='Actual Inflow'
                ))
                
                fig_forecast.update_layout(
                    title=f"12-Week Inflow Forecast for {selected_res_name} (Sample #{batch_num})",
                    xaxis_title="Weeks Ahead",
                    yaxis_title="Inflow (MCM)",
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig_forecast, use_container_width=True)
                
    except Exception as e:
        st.error(f"Error loading model/data: {e}")
        st.info("Ensure the model is trained and `main.py` is fully functional.")

# ==========================================
# TAB 3: EXPLAINABILITY (XAI)
# ==========================================
with tab3:
    st.header("Explainability (XAI)")
    st.markdown("Understand what the GNN-ReservoirNet is looking at when making its predictions.")
    
    xai_data = load_json(XAI_REPORT_PATH)
    
    if xai_data is not None:
        col_feats, col_attn = st.columns(2)
        
        with col_feats:
            st.subheader("Feature Importance (Integrated Gradients)")
            
            node_feats = xai_data.get("feature_importance", {}).get("node_features", {})
            clim_feats = xai_data.get("feature_importance", {}).get("climate_features", {})
            
            # Combine them for visualization
            all_feats = {**node_feats, **clim_feats}
            feat_df = pd.DataFrame(list(all_feats.items()), columns=["Feature", "Importance"])
            feat_df = feat_df.sort_values(by="Importance", ascending=True) # Ascending for horizontal bar
            
            fig_feats = px.bar(
                feat_df,
                x="Importance",
                y="Feature",
                orientation='h',
                title="Global Attribution Scores",
                color="Importance",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig_feats, use_container_width=True)
            
        with col_attn:
            st.subheader("Spatial Attention (GAT)")
            
            attn_data = xai_data.get("spatial_attention", {})
            
            # Build networkx graph
            G = nx.DiGraph()
            for edge, weight in attn_data.items():
                src, dst = edge.split("_")
                G.add_edge(f"R{int(src)+1:02d}", f"R{int(dst)+1:02d}", weight=weight)
                
            pos = nx.spring_layout(G, seed=42)
            
            edge_x = []
            edge_y = []
            edge_weights = []
            
            for src, dst, data in G.edges(data=True):
                x0, y0 = pos[src]
                x1, y1 = pos[dst]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
                edge_weights.append(data['weight'])
                
            # Node trace
            node_x = []
            node_y = []
            node_text = []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
                
            node_trace = go.Scatter(
                x=node_x, y=node_y,
                mode='markers+text',
                text=node_text,
                textposition="bottom center",
                hoverinfo='text',
                marker=dict(size=20, color='LightSkyBlue', line=dict(width=2, color='DarkSlateGrey'))
            )
            
            # We plot edges individually to vary thickness based on attention weight
            fig_attn = go.Figure()
            for i, (src, dst, data) in enumerate(G.edges(data=True)):
                x0, y0 = pos[src]
                x1, y1 = pos[dst]
                w = data['weight']
                fig_attn.add_trace(go.Scatter(
                    x=[x0, x1], y=[y0, y1],
                    mode='lines',
                    line=dict(width=max(0.5, w * 10), color='rgba(255, 100, 100, 0.8)'),
                    hoverinfo='text',
                    text=f"{src} -> {dst}: {w:.4f}"
                ))
                
            fig_attn.add_trace(node_trace)
            
            fig_attn.update_layout(
                title="GAT Attention Flow",
                showlegend=False,
                hovermode='closest',
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            st.plotly_chart(fig_attn, use_container_width=True)
            
    else:
        st.warning("Explainability report not found. Run `main.py --explain` first.")
