# GNN-ReservoirNet: Spatiotemporal Prediction of Reservoir Levels in Peninsular India during El Niño

## Overview
GNN-ReservoirNet is a hybrid physics-informed machine learning architecture designed to predict multi-step ahead reservoir levels and inflows for Peninsular India, explicitly conditioned on global climate phenomena like El Niño-Southern Oscillation (ENSO). 

It features a **two-stage architecture**:
1. **GNN Inflow Prediction:** Uses Graph Attention Networks (GAT) to model physical and climatological relationships, combined with Temporal Convolutional Networks (TCN) for temporal dynamics. A Cross-Attention mechanism is used to incorporate ENSO and Indian Ocean Dipole (IOD) anomalies.
2. **Mass-Balance Storage:** A deterministic physical layer computes reservoir storage iteratively using the predicted inflow:
   `S(t+1) = S(t) + I(t) - E(t) - R(t)`

## Key Features
- **Spatial Modeling (GAT):** Graph Attention Networks operating over physical connectivity and climatological correlation edges.
- **Temporal Modeling (TCN):** 90-day lookback window for capturing lagged relationships in hydrometeorological data.
- **Climate Context (Cross-Attention):** explicitly models the interaction between local features and global climate indices (ENSO/IOD).
- **Quantile Regression Loss:** Direct multi-step quantile predictions (P10/P50/P90) with pinball loss and ONI-conditioned asymmetric penalty for extreme events.
- **Explainability:** Employs Integrated Gradients and Spatial Attention Maps for model transparency and interpretability.

## Installation
Requires Python 3.10+.
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Project Structure
```
PBL/
├── configs/             # Configuration files (YAML)
├── src/
│   ├── data/            # Data loaders and preprocessing
│   ├── models/          # GNN, TCN, Mass-Balance implementations
│   ├── training/        # Training loops and loss functions
│   ├── evaluation/      # Metrics and evaluation scripts
│   ├── explainability/  # Explanability modules (Captum, etc.)
│   ├── inference/       # Inference and ONNX export scripts
│   └── utils/           # Utilities and helpers
├── requirements.txt
└── README.md
```

## Quick Start
1. Configure your data directories in `configs/default_config.yaml`.
2. Define the reservoir network in `configs/reservoirs.yaml`.
3. Pretrain and fine-tune the model:
   ```bash
   python -m src.training.train --config configs/default_config.yaml
   ```
4. Evaluate and generate explainability maps:
   ```bash
   python -m src.evaluation.eval --config configs/default_config.yaml
   ```

## Data Sources
- **Reservoir Levels:** India Water Resources Information System (WRIS).
- **Meteorology:** IMD (Indian Meteorological Department) or ERA5.
- **Climate Indices:** NOAA (ENSO/ONI, SOI) and BOM (IOD).
