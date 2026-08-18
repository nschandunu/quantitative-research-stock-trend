# AAPL Stock Trend Prediction with TCN

A quantitative research project for predicting the daily short-term trend of Apple Inc. (AAPL) using a Temporal Convolutional Network (TCN) and evaluating the resulting signals through backtesting.

## Overview

The project follows a chronological, research-oriented workflow:

1. Load one year of daily AAPL OHLCV data.
2. Calculate five technical indicators.
3. Generate binary Rise/Fall trend labels using the assessment labeling methodology.
4. Construct 22-day supervised learning windows.
5. Split the data chronologically into training, validation, and test sets.
6. Fit feature scaling using training data only.
7. Train a TCN classifier.
8. Evaluate the model on the unseen test set.
9. Compare performance against a majority-class baseline.
10. Run a full-year signal backtest and compare it with Buy & Hold.
11. Provide a standalone `main.py` inference pipeline using the saved model and scaler.

## Features

The model uses nine input features:

- Open
- High
- Low
- Close
- ATR
- EMA20
- MOM6
- CCI
- MACD

The model input is a **22-day × 9-feature** sequence.

## Model

The prediction model is a Temporal Convolutional Network consisting of:

- Temporal Block 1: 32 channels, kernel size 3, dilation 1
- Temporal Block 2: 32 channels, kernel size 3, dilation 2
- Dropout: 0.2
- ReLU activations
- Final linear classifier
- Two output classes: `Fall` and `Rise`

The trained artifacts are stored in:

```text
models/
├── aapl_tcn.pt
└── aapl_scaler.joblib
```

## Project Structure

```text
quantitative-research-stock-trend/
├── data/
│   └── AAPL_1y_daily.csv
├── models/
│   ├── aapl_tcn.pt
│   └── aapl_scaler.joblib
├── notebooks/
│   ├── 01_*.ipynb
│   ├── 02_*.ipynb
│   ├── 03_dataset_split_and_scaling.ipynb
│   ├── 04_*.ipynb
│   └── 05_backtesting.ipynb
├── main.py
└── README.md
```

## Requirements

Python environment with the required packages installed, including:

- NumPy
- pandas
- PyTorch
- scikit-learn
- joblib
- backtesting

## Running the Prediction Pipeline

From the project root:

```bash
python main.py
```

The script:

1. Loads `data/AAPL_1y_daily.csv`.
2. Calculates the five technical indicators.
3. Builds the latest 22-day model input.
4. Loads the training-only scaler from `models/aapl_scaler.joblib`.
5. Scales the input.
6. Loads the trained TCN from `models/aapl_tcn.pt`.
7. Generates the latest Rise/Fall prediction.
8. Reports the class probabilities and confidence.

Example output format:

```text
============================================================
AAPL TCN PREDICTION RESULT
============================================================
Prediction date: 2026-08-17
Signal: Fall
Fall probability: 0.9516
Rise probability: 0.0484
Confidence: 95.16%
============================================================
Prediction complete.
============================================================
```

## Model Evaluation

The final unseen test set contains **42 samples**.

### Test Performance

| Metric | Result |
|---|---:|
| Accuracy | 42.86% |
| Balanced Accuracy | 50.00% |
| Macro F1 | 0.30 |

The TCN did **not** outperform the majority-class baseline on the held-out test set.

### Baseline

| Metric | Baseline | TCN |
|---|---:|---:|
| Accuracy | 57.14% | 42.86% |
| Balanced Accuracy | 50.00% | 50.00% |

The baseline therefore outperformed the TCN on test accuracy.

## Full-Year Backtest

After applying the model's 22-day lookback requirement, the backtest contains **211 prediction days** covering:

```text
2025-10-14 → 2026-08-17
```

The trading strategy buys/holds when the model predicts `Rise` and remains out of the position when the model predicts `Fall`.

### Backtest Results

| Metric | Result |
|---|---:|
| Strategy Return | +30.05% |
| Buy & Hold Return | +22.95% |
| Outperformance | +7.10 percentage points |
| Number of Trades | 8 |

The implemented trading strategy outperformed Buy & Hold over the evaluated backtest period.

## Important Interpretation

The model's classification performance and the trading strategy's historical backtest performance should be interpreted separately.

The held-out test results show that the TCN did not beat the majority-class baseline. The full-year backtest nevertheless produced a higher historical return than Buy & Hold under the implemented trading rules.

The backtest result should therefore **not** be interpreted as proof that the model will outperform the market in future periods.

## Reproducibility

The project uses:

- Chronological train/validation/test splitting.
- Training-only feature scaling.
- A saved model checkpoint.
- A saved training scaler.
- A fixed 22-day sequence length.
- The same nine model features throughout the workflow.

The notebooks contain validation checks for data preparation, feature construction, sequence generation, scaling, model inference, test evaluation, and backtesting.

## Main Files

### `03_dataset_split_and_scaling.ipynb`

Creates the chronological train/validation/test datasets and performs training-only scaling.

### `05_backtesting.ipynb`

Loads the trained model and scaler, generates predictions, evaluates the test set, and performs the full-year backtest.

### `main.py`

Provides a standalone inference pipeline using the saved model and scaler.

## Assessment Deliverables

The project includes:

- Source code and notebooks
- Trained TCN model
- Saved feature scaler
- Standalone prediction script
- Test-set evaluation
- Baseline comparison
- Full-year backtesting
- Buy & Hold comparison
