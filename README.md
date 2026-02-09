# Time-series counterfactual inference with penalized autoencoders

Code for for the TMLR paper:

> **CEPAE: Conditional Entropy-Penalized Autoencoders for Time Series Counterfactuals**

This repository contains:
- Autoencoder-based counterfactual generators (**CEPAE**, **CVAE**, **CAAE**)
- Forecasting baselines (**LSTM forecast**, **adversarially balanced forecast / AB-LSTM**)
- Reproducible notebooks for semi-synthetic (Rossmann) and synthetic experiments
- A small set of evaluation metrics (MAE/MBE, added-variations, axiomatic metrics)

## Repository structure

- `src/`
  - `models/` – model definitions and training utilities
  - `data/` – dataset loaders / synthetic generators
  - `metrics/` – evaluation metrics used in the notebooks
  - `utils/` – small helpers (schedules)
- `notebooks/`
  - `Rossmann.ipynb` – semi-synthetic experiments (requires Kaggle data)
  - `confounded.ipynb` – synthetic confounded setting
  - `unconfounded.ipynb` – synthetic unconfounded setting
- `data/` – put datasets here 

## Setup

Create an environment (Python ≥ 3.9 recommended) and install dependencies:

```bash
pip install -r requirements.txt
```

To run the notebooks:

```bash
pip install jupyterlab
jupyter lab
```

## Rossmann data

Download `train.csv` from the Kaggle competition **Rossmann Store Sales** (https://www.kaggle.com/competitions/rossmann-store-sales/data) and place it at:

```
data/rossmann/train.csv
```

Then run `notebooks/Rossmann.ipynb`.

