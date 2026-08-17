# Short-Term Stock Price Trend Prediction

Research-based implementation of:

> Short-Term Stock Price-Trend Prediction Using Meta-Learning

## Objective

Build a short-term stock trend prediction model using daily OHLC data and five technical indicators from the referenced research paper.

The trained model will be integrated with `backtesting.py` to evaluate a simple Rise/Fall trading strategy.

## Asset

AAPL - Apple Inc.

## Project Structure

- `data/` - Raw and processed datasets
- `notebooks/` - Research and experimentation
- `src/` - Reusable Python modules
- `models/` - Trained models
- `backtest/` - backtesting.py implementation
- `results/` - Experimental results
- `reports/` - Final report
- `assets/` - Supporting/demo assets

## Status

- [ ] Environment setup
- [ ] Data collection
- [ ] Technical indicators
- [ ] Trend labeling
- [ ] Dataset construction
- [ ] TCN model
- [ ] Model training
- [ ] Model evaluation
- [ ] backtesting.py integration
- [ ] Strategy evaluation
- [ ] Report
- [ ] Demo video