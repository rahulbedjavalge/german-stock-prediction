# German Stock Market Prediction Models

This repository contains machine learning models for predicting German stock prices and other ML examples.

## 📁 Files

- **`german_stock_prediction.py`** - Advanced stock market prediction model for German DAX stocks
- **`linear_regression.py`** - Linear regression example and tutorial
- **`hi.py`** - Logistic regression example and tutorial

## 🎯 German Stock Market Predictor

A comprehensive machine learning model that predicts prices for major German stocks (DAX components).

### Features:
- ✅ Real-time data fetching from Yahoo Finance
- ✅ Advanced technical indicators (RSI, MACD, Bollinger Bands, Moving Averages)
- ✅ Random Forest Regressor for accurate predictions
- ✅ Future price predictions (5 days ahead)
- ✅ Beautiful visualizations and performance metrics

### Supported Stocks:
- SAP (SAP.DE)
- Siemens (SIE.DE)
- Volkswagen (VOW3.DE)
- Allianz (ALV.DE)
- Deutsche Bank (DBK.DE)
- BMW (BMW.DE)
- Adidas (ADS.DE)
- BASF (BAS.DE)
- Bayer (BAYN.DE)
- Mercedes-Benz (MBG.DE)

### Usage:
```bash
python german_stock_prediction.py
```

### Model Performance:
- R² Score: ~80% on test data
- MAPE: ~2% (Mean Absolute Percentage Error)
- RMSE: ~€5 for SAP stock

## 📦 Installation

Install required packages:
```bash
pip install yfinance pandas numpy scikit-learn matplotlib
```

## 🚀 Getting Started

1. Clone the repository
2. Install dependencies
3. Run any of the prediction models
4. View results and visualizations

## 📊 Technical Indicators Used

- **Moving Averages (MA)**: 5, 10, 20, 50-day periods
- **Exponential Moving Averages (EMA)**: 12, 26-day periods
- **MACD**: Moving Average Convergence Divergence
- **RSI**: Relative Strength Index (14-day)
- **Bollinger Bands**: 20-day with 2σ
- **ROC**: Rate of Change
- **Volume Analysis**: Volume changes and trends

## 📈 Prediction Output

The model provides:
- Next 5 days price predictions
- Trend analysis (Bullish/Bearish)
- Model performance metrics
- Visual plots comparing actual vs predicted prices

## ⚠️ Disclaimer

This is an educational project. Stock market predictions are not guaranteed and should not be used as financial advice. Always do your own research before making investment decisions.

## 📝 License

MIT License - Feel free to use and modify for your projects!
