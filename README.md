<div align="center">
                          
# LSTM Time-Series Forecasting
 
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-orange)
![Metrics](https://img.shields.io/badge/Metrics-RMSE%20%2B%20MAE%20%2B%20MAPE-green)
![Status](https://img.shields.io/badge/Status-Portfolio%20Project-purple)
[![CI](https://github.com/AmirhosseinHonardoust/LSTM-Time-Series-Forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/AmirhosseinHonardoust/LSTM-Time-Series-Forecasting/actions/workflows/ci.yml)

</div>

A hands-on project for forecasting time-series with **PyTorch LSTMs**. It creates realistic daily data (trend, seasonality, events, noise), prepares it with **sliding windows**, and trains an **LSTM** to make **multi-step predictions**. The project tracks errors with **RMSE, MAE, MAPE** and shows clear plots of training progress and forecast results.

> **Note on "forecast":** the plots and metrics in this README evaluate the model on the last `horizon` days already present in the input CSV (a **backtest**), not real future dates beyond it, the "actual" values plotted alongside the forecast are ground truth, not projections.
>
> To forecast beyond the CSV's date range, extend `data/daily_series.csv` (or generate a longer series) first.

---

## Table of Contents

- [Project Overview](#project-overview)
- [What This Project Does](#what-this-project-does)
- [What This Project Does Not Do](#what-this-project-does-not-do)
- [Key Features](#key-features)
- [System Workflow](#system-workflow)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Generating Data](#generating-data)
- [Training the Model](#training-the-model)
- [Evaluation](#evaluation)
- [Results](#results)
- [Visual Reports](#visual-reports)
- [Model Artifacts](#model-artifacts)
- [Testing and CI](#testing-and-ci)
- [Code Quality](#code-quality)
- [Limitations](#limitations)
- [Future Improvements](#future-improvements)
- [Tech Stack](#tech-stack)
- [Author](#author)
- [License](#license)

---

## Project Overview

Time-series forecasting is often demonstrated with a single loss curve and a plot that looks convincing without much explanation of what is actually being measured. A forecast is only useful if it is clear what data it was evaluated on:

- generate a realistic, reproducible daily series instead of relying on a single fixed dataset
- prepare it with sliding windows for supervised learning
- train an LSTM with dropout, Adam, and early stopping
- predict a full horizon in one forward pass (direct multi-step forecasting)
- report RMSE, MAE, and MAPE, and be explicit about backtest vs. true future forecasting

This project demonstrates an end-to-end PyTorch forecasting workflow: synthetic data generation, sliding-window dataset preparation, LSTM training with early stopping, evaluation with standard error metrics, and clear visualizations of both training progress and forecast quality.

The goal is to show a clean, reproducible time-series forecasting pipeline that goes from raw dates to trained model to evaluated forecast, not just a single plot.

---

## What This Project Does

This project can:

- Generate a synthetic daily time series with trend, seasonality, events, and noise
- Prepare sliding-window sequences for supervised learning
- Train a PyTorch LSTM with dropout, the Adam optimizer, and early stopping
- Forecast a full horizon in a single forward pass (direct multi-step forecasting)
- Compute RMSE, MAE, and MAPE on a held-out portion of the series
- Reconstruct the exact trained architecture at evaluation time from the saved checkpoint
- Generate training/validation loss curves and a forecast-vs-actual plot
- Save model artifacts: `best_lstm.pt`, `scaler.json`, `metrics.json`
- Run automated tests and a CI workflow on every push and pull request

---

## What This Project Does Not Do

This project does **not**:

- Forecast real future dates beyond the input CSV's date range without first extending the data
- Fetch or use real-world market, weather, or sensor data by default
- Guarantee that error metrics on synthetic data transfer to real-world series
- Perform hyperparameter search or model selection automatically
- Support multivariate inputs or exogenous regressors out of the box

A production forecasting system would need real source data, exogenous features, hyperparameter tuning, and ongoing retraining/monitoring as the underlying process drifts.

---

## Key Features

- **Synthetic daily series generation** with configurable length and seed
- **Sliding-window dataset preparation** for supervised learning
- **LSTM model** with dropout, Adam optimizer, and early stopping
- **Direct multi-step forecasting**, one forward pass predicts the whole horizon
- **RMSE, MAE, MAPE** evaluation metrics
- **Training curves and forecast plots** saved automatically
- **Checkpoint-embedded architecture**, `evaluate.py` reconstructs the same model from `best_lstm.pt` with no need to re-specify architecture flags
- **Configurable architecture**: hidden size, number of layers, dropout, and early-stopping patience
- **Unit tests and GitHub Actions CI**
- **Ruff, Black, and mypy** configured for code quality

---

## System Workflow

```text
Synthetic daily series (trend + seasonality + events + noise)
        ↓
Sliding-window dataset preparation
        ↓
LSTM model (dropout, Adam, early stopping)
        ↓
Direct multi-step forecast (single forward pass)
        ↓
RMSE / MAE / MAPE on held-out horizon
        ↓
Training curves and forecast-vs-actual plot
        ↓
Checksum-free, checkpoint-embedded artifacts (best_lstm.pt, scaler.json, metrics.json)
```

---

## Project Structure

```text
lstm-time-series-forecasting/
├── README.md
├── LICENSE
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── .github/
│   └── workflows/
│       └── ci.yml
├── data/
│   └── generate_series.py
├── src/
│   ├── model.py
│   ├── scaler.py
│   ├── train_lstm.py
│   ├── evaluate.py
│   └── utils.py
├── tests/
│   ├── test_evaluate_config.py
│   ├── test_generate_series.py
│   ├── test_scaler.py
│   ├── test_train_eval_smoke.py
│   └── test_utils.py
└── outputs/
    └── (generated by running the scripts below, not committed)
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/AmirhosseinHonardoust/LSTM-Time-Series-Forecasting.git
cd LSTM-Time-Series-Forecasting
```

### 2. Create a Virtual Environment

On Windows CMD:

```cmd
python -m venv .venv
.venv\Scripts\activate
```

On macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

For development tools (pytest, Ruff, Black, mypy):

```bash
pip install -r requirements-dev.txt
```

Optionally, run the same checks automatically on every commit:

```bash
pip install pre-commit
pre-commit install
```

---

## Quick Start

Generate the data:

```bash
python data/generate_series.py --start 2020-01-01 --end 2025-12-31 --seed 42 --out data/daily_series.csv
```

Train the model:

```bash
python src/train_lstm.py --input data/daily_series.csv --horizon 30 --lookback 60 --epochs 30 --batch-size 64 --outdir outputs --seed 42
```

Evaluate it:

```bash
python src/evaluate.py --input data/daily_series.csv --model outputs/best_lstm.pt --outdir outputs
```

---

## Generating Data

`data/daily_series.csv` is committed as a ready-to-use sample, it is exactly the output of the command below (seed 42), kept in the repo so `train_lstm.py` and `evaluate.py` work without a generation step first.

```bash
python data/generate_series.py --start 2020-01-01 --end 2025-12-31 --seed 42 --out data/daily_series.csv
```

Re-run the command (with a different `--seed` or date range) to regenerate or replace it.

---

## Training the Model

```bash
python src/train_lstm.py \
  --input data/daily_series.csv \
  --horizon 30 \
  --lookback 60 \
  --epochs 30 \
  --batch-size 64 \
  --outdir outputs \
  --seed 42
```

Architecture and early stopping are also configurable:

<div align="center">

| Flag | Default | Purpose |
|---|---|---|
| `--hidden-size` | 64 | LSTM hidden state size |
| `--num-layers` | 2 | Number of stacked LSTM layers |
| `--dropout` | 0.2 | Dropout applied between LSTM layers |
| `--patience` | 5 epochs | Early-stopping patience on validation loss |
| `--device` | `auto` | `auto` picks CUDA, then Apple MPS, then CPU. Or set explicitly: `cpu`, `cuda`, `cuda:0`, `mps` |
| `--date-col` | `date` | Date column name in `--input`, for CSVs that don't use the bundled generator's naming |
| `--value-col` | `value` | Value column name in `--input` |

</div>

Whatever you choose is saved into `outputs/best_lstm.pt` alongside the weights, so `evaluate.py` always reconstructs the exact same model, no need to re-specify architecture flags at evaluation time.

---

## Evaluation

```bash
python src/evaluate.py \
  --input data/daily_series.csv \
  --model outputs/best_lstm.pt \
  --outdir outputs
```

`--lookback`/`--horizon` are optional here and default to whatever the checkpoint was trained with. Pass them explicitly only to double-check they match, a mismatched value raises a clear error instead of a raw tensor shape-mismatch failure.

`--device`, `--date-col`, and `--value-col` work the same way as in `train_lstm.py` (see above); a CSV missing either configured column raises a clear error instead of a raw `KeyError`.

---

## Results

<div align="center">

| Metric | Value |
|---|---|
| RMSE | 22.71 |
| MAE | 16.74 |
| MAPE | 7.96% |

</div>

These numbers come from the included run and are a **backtest** on the last `horizon` days of `data/daily_series.csv`, evaluated against ground truth already present in that file, not a prediction of unseen future dates.

---

## Visual Reports

### Forecast and training behavior

<div align="center">

| Forecast vs Actual | Training & Validation Loss |
|---|---|
|<img width="1600" height="800" alt="forecast_plot" src="https://github.com/user-attachments/assets/a0cf878f-4ef4-47d6-9911-817ea947fd87" /> | <img width="1120" height="800" alt="training_curves" src="https://github.com/user-attachments/assets/374b5e88-b793-4214-9315-1750226703db" /> |
| **Analysis:** The plot compares the model's direct multi-step forecast against the ground-truth values for the same held-out horizon. Because both series come from the same input CSV, this shows backtest quality, not live forecasting performance. | **Analysis:** The training/validation curves show whether the model is still improving when early stopping triggers, which is a quick check against both underfitting and overfitting. |

</div>

---

## Model Artifacts

Training writes three files to `outputs/`:

<div align="center">

| Artifact | Contents |
|---|---|
| `outputs/best_lstm.pt` | Trained PyTorch weights, plus the architecture config used to train them |
| `outputs/scaler.json` | Fitted scaler (mean/std), stored as plain JSON |
| `outputs/metrics.json` | RMSE, MAE, MAPE from evaluation |

</div>

Because the architecture config travels with the checkpoint, `evaluate.py` rebuilds the exact same model automatically, you don't need to remember or re-pass `--hidden-size`, `--num-layers`, or `--dropout` at evaluation time.

---

## Testing and CI

Run unit tests locally:

```bash
pytest
```

Run the same checks CI runs:

```bash
ruff check .
black --check .
mypy src data
pytest
```

CI (`.github/workflows/ci.yml`) runs these checks on every push and pull request. It also runs a `pip-audit` dependency scan (informational, `continue-on-error`); pinned dependency versions are kept on the latest audit-clean release available at time of update. Contributing changes? See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Code Quality

The project separates responsibilities across modules:

<div align="center">

| Module | Purpose |
|---|---|
| `data/generate_series.py` | Generates the synthetic daily series (trend, seasonality, events, noise) |
| `src/model.py` | LSTM model definition |
| `src/scaler.py` | Fitting/applying the feature scaler |
| `src/train_lstm.py` | Sliding-window dataset prep, training loop, early stopping, artifact saving |
| `src/evaluate.py` | Loads a checkpoint, reconstructs the model, computes RMSE/MAE/MAPE, generates plots |
| `src/utils.py` | Shared helpers |

</div>

**On imports:** `src/` and `data/` are plain script directories, not installed packages, modules import each other with bare names (`from model import ...`) rather than `from src.model import ...`. This works because Python adds a script's own directory to `sys.path` when it's run directly (`tests/conftest.py` does the same for `pytest`). This is a deliberate choice to keep the project runnable with just `pip install -r requirements.txt` and no editable install; it's not meant to be imported as a library from outside `src/`/`data/`.

Tooling is configured through `pyproject.toml` (Ruff, Black, mypy, pytest) and `requirements-dev.txt`.

---

## Limitations

This project has important limitations:

- The bundled series is synthetic, not real-world market/weather/sensor data
- Reported metrics are a backtest on the same CSV, not validated future performance
- There is no hyperparameter search, defaults are reasonable, not tuned
- The model is univariate; it does not use exogenous regressors or external features
- Forecasting beyond the CSV's date range requires extending the data first

The project is strongest as a portfolio demonstration of a clean, reproducible PyTorch LSTM forecasting pipeline.

---

## Future Improvements

Potential next improvements:

- Add support for real-world datasets (energy, retail, weather) alongside the synthetic generator
- Add exogenous/multivariate inputs
- Add hyperparameter search (e.g. Optuna) instead of fixed defaults
- Add rolling-origin cross-validation instead of a single holdout split
- Add confidence intervals / probabilistic forecasting
- Add Docker support
- Explore attention-based or Transformer forecasting baselines for comparison

---

## Tech Stack

- Python
- PyTorch
- NumPy / pandas
- matplotlib
- pytest
- Ruff, Black, mypy
- GitHub Actions

---

## Author

**Amir Honardoust**

GitHub: [@AmirhosseinHonardoust](https://github.com/AmirhosseinHonardoust)

---

## License

This project is licensed under the MIT License. See [`LICENSE`](LICENSE) for details.
