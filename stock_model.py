import os
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

try:
    import yfinance as yf
except Exception:
    yf = None


def fetch_data(ticker, period='1y'):
    if yf is None:
        raise RuntimeError('yfinance is required to fetch data. Install via `pip install yfinance`')
    df = yf.download(ticker, period=period, progress=False)
    if df.empty:
        raise ValueError(f'No data fetched for {ticker}')
    return df


def create_lag_features(df, n_lags=5):
    df = df.copy()
    # Ensure column names are simple strings (yfinance can sometimes return tuple column names)
    df.columns = [c if isinstance(c, str) else c[0] if isinstance(c, tuple) else str(c) for c in df.columns]
    for i in range(1, n_lags + 1):
        df[f'Close_lag_{i}'] = df['Close'].shift(i)
    if 'Close' not in df.columns:
        # fallback: try to find a column that contains 'Close'
        close_cols = [c for c in df.columns if 'close' in c.lower()]
        if not close_cols:
            raise KeyError('Close column not found in data')
        df['Close'] = df[close_cols[0]]
    df['Target'] = df['Close'].shift(-1)
    df = df.dropna()
    return df


def train_model(df, n_estimators=50, test_size=0.2, random_state=42):
    feature_cols = [c for c in df.columns if c.startswith('Close_lag_')]
    X = df[feature_cols].values
    y = df['Target'].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics = {'rmse': float(rmse), 'r2': float(r2), 'train_size': len(X_train), 'test_size': len(X_test)}

    return model, metrics


def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def main(args):
    print(f"Fetching data for {args.ticker} ({args.period})...")
    df = fetch_data(args.ticker, period=args.period)
    df_feats = create_lag_features(df, n_lags=args.n_lags)

    print(f"Training RandomForest (n_estimators={args.n_estimators})...")
    model, metrics = train_model(df_feats, n_estimators=args.n_estimators, test_size=args.test_size)

    print(f"Training complete — RMSE: {metrics['rmse']:.4f}, R2: {metrics['r2']:.4f}")

    save_path = args.save_path or f"models/{args.ticker}_rf.joblib"
    save_model(model, save_path)
    print(f"Model saved to: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a simple stock RandomForest model using yfinance data')
    parser.add_argument('--ticker', type=str, default='AAPL')
    parser.add_argument('--period', type=str, default='1y')
    parser.add_argument('--n_lags', type=int, default=5)
    parser.add_argument('--n_estimators', type=int, default=50)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--save_path', type=str, default='models/stock_model.joblib')

    args = parser.parse_args()
    main(args)
