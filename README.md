# Coastal Groundwater Level Prediction using Random Forest

This repository contains the code associated with the paper. It trains a Random Forest model to predict median coastal groundwater levels (GWL) across New Zealand and applies it under two IPCC climate scenarios (SSP2-4.5 and SSP5-8.5) to estimate rainfall-driven changes by 2080–2099.

## Repository structure

```
.
├── 01_model_training.ipynb      # Model training, evaluation, and baseline spatial prediction
├── 02_scenario_analysis.ipynb   # Future GWL prediction under SSP2-4.5 and SSP5-8.5
├── utils.py                     # Helper functions for sampling precipitation change from NetCDF
└── requirements.txt
```

## Data availability

The data used in this study are not publicly available due to third-party licensing restrictions. Readers interested in accessing the data may contact the corresponding author.

The notebooks expect a `data/` folder in the working directory containing the following files:

| File | Description |
|------|-------------|
| `Coastal_Komanawa_BoreParams_Mar2025(in).csv` | Training dataset of coastal well observations |
| `modelGrid_March2025.csv` | National 250 m model grid with predictor variables |
| `PR_ssp245_MMM_CCAM_change_fp2080-2099_bp1986-2005_ANN_NZ5km.nc` | CCAM-downscaled precipitation change (SSP2-4.5) |
| `PR_ssp585_MMM_CCAM_change_fp2080-2099_bp1986-2005_ANN_NZ5km.nc` | CCAM-downscaled precipitation change (SSP5-8.5) |

Precipitation change NetCDF files contain multi-model mean (MMM) annual percentage change relative to the 1986–2005 baseline on a 5 km WGS84 grid.

## How to run

Run the notebooks in order:

1. **`01_model_training.ipynb`** — loads training data, tunes hyperparameters via 7-fold cross-validated grid search, evaluates on a held-out 20% test set, and saves the fitted model to `rf_groundwater_model.joblib`. Also generates a baseline spatial prediction over the national model grid.

2. **`02_scenario_analysis.ipynb`** — loads the saved model and applies it under SSP2-4.5 and SSP5-8.5 precipitation projections. Outputs per-scenario GWL predictions and computes ΔGWL (future − baseline).

## Requirements

Python 3.10 is recommended. Install dependencies with:

```bash
pip install -r requirements.txt
```

## Outputs

| File | Description |
|------|-------------|
| `rf_groundwater_model.joblib` | Trained Random Forest model |
| `data/Predicted_GWL_baseline.csv` | Baseline GWL predictions across the national grid |
| `data/Predicted_GWL_ssp245.csv` | GWL predictions under SSP2-4.5 |
| `data/Predicted_GWL_ssp585.csv` | GWL predictions under SSP5-8.5 |
