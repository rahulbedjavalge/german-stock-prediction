import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================
# DUAL MODEL STOCK PREDICTION - 10 YEARS DATA
# ============================================

class DualModelStockPredictor:
    """
    Compare Linear Regression vs Random Forest for stock price prediction
    Using 10 years of historical data
    """
    
    def __init__(self, ticker, period='10y'):
        """
        Initialize the dual model predictor
        
        Parameters:
        - ticker: Stock ticker symbol
        - period: Historical data period (default: 10 years)
        """
        self.ticker = ticker
        self.period = period
        self.data = None
        self.model_lr = None
        self.model_rf = None
        self.scaler = MinMaxScaler()
        
    def fetch_data(self):
        """Fetch 10 years of stock data"""
        print(f"\n{'=' * 70}")
        print(f"FETCHING 10 YEARS OF DATA FOR {self.ticker}")
        print(f"{'=' * 70}")
        
        stock = yf.Ticker(self.ticker)
        self.data = stock.history(period=self.period)
        
        if self.data.empty:
            raise ValueError(f"No data found for ticker {self.ticker}")
        
        print(f"✓ Successfully fetched {len(self.data)} days of data")
        print(f"✓ Date range: {self.data.index[0].date()} to {self.data.index[-1].date()}")
        print(f"✓ Years of data: {(self.data.index[-1] - self.data.index[0]).days / 365.25:.1f} years")
        return self.data
    
    def create_features(self, data):
        """Create technical indicators and features"""
        df = data.copy()
        
        # Moving Averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        df['MA_200'] = df['Close'].rolling(window=200).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        bb_std = df['Close'].rolling(window=20).std()
        df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
        df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
        df['BB_width'] = df['BB_upper'] - df['BB_lower']
        
        # ATR (Average True Range)
        df['TR'] = np.maximum(
            df['High'] - df['Low'],
            np.maximum(
                abs(df['High'] - df['Close'].shift()),
                abs(df['Low'] - df['Close'].shift())
            )
        )
        df['ATR'] = df['TR'].rolling(window=14).mean()
        
        # ROC (Rate of Change)
        df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        
        # Stochastic Oscillator
        low_min = df['Low'].rolling(window=14).min()
        high_max = df['High'].rolling(window=14).max()
        df['Stoch_K'] = (df['Close'] - low_min) / (high_max - low_min) * 100
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # Volume features
        df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
        df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Price changes
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Diff'] = df['High'] - df['Low']
        df['High_Low_Pct'] = ((df['High'] - df['Low']) / df['Close']) * 100
        df['Open_Close_Diff'] = df['Close'] - df['Open']
        df['Close_Position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'])
        
        # Lag features
        for i in range(1, 8):
            df[f'Close_Lag_{i}'] = df['Close'].shift(i)
            df[f'Volume_Lag_{i}'] = df['Volume'].shift(i)
        
        # Target: Next day closing price
        df['Target'] = df['Close'].shift(-1)
        
        # Drop NaN values
        df = df.dropna()
        
        # Replace infinity values
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        return df
    
    def prepare_data(self):
        """Prepare data for training"""
        print(f"\n{'=' * 70}")
        print("CREATING FEATURES AND PREPARING DATA")
        print(f"{'=' * 70}")
        
        df = self.create_features(self.data)
        
        # Select all feature columns
        feature_columns = [col for col in df.columns if col != 'Target']
        
        X = df[feature_columns].values
        y = df['Target'].values
        
        # Split: 80% train, 20% test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✓ Total features created: {len(feature_columns)}")
        print(f"✓ Training samples: {len(X_train)} ({len(X_train)/(len(X_train)+len(X_test))*100:.1f}%)")
        print(f"✓ Testing samples: {len(X_test)} ({len(X_test)/(len(X_train)+len(X_test))*100:.1f}%)")
        print(f"✓ Training period: {df.index[0].date()} to {df.index[len(X_train)].date()}")
        print(f"✓ Testing period: {df.index[len(X_train)].date()} to {df.index[-1].date()}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, df, feature_columns
    
    def train_models(self):
        """Train both Linear Regression and Random Forest models"""
        X_train, X_test, y_train, y_test, df, feature_cols = self.prepare_data()
        
        print(f"\n{'=' * 70}")
        print("TRAINING MODELS")
        print(f"{'=' * 70}")
        
        # ===== LINEAR REGRESSION =====
        print("\n📊 Training Linear Regression...")
        self.model_lr = LinearRegression()
        self.model_lr.fit(X_train, y_train)
        
        y_pred_lr_train = self.model_lr.predict(X_train)
        y_pred_lr_test = self.model_lr.predict(X_test)
        
        lr_train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_lr_train))
        lr_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_lr_test))
        lr_train_r2 = r2_score(y_train, y_pred_lr_train)
        lr_test_r2 = r2_score(y_test, y_pred_lr_test)
        lr_train_mape = mean_absolute_percentage_error(y_train, y_pred_lr_train) * 100
        lr_test_mape = mean_absolute_percentage_error(y_test, y_pred_lr_test) * 100
        
        print("✓ Linear Regression trained!")
        
        # ===== RANDOM FOREST =====
        print("\n🌲 Training Random Forest (this may take a minute)...")
        self.model_rf = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        )
        self.model_rf.fit(X_train, y_train)
        
        y_pred_rf_train = self.model_rf.predict(X_train)
        y_pred_rf_test = self.model_rf.predict(X_test)
        
        rf_train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_rf_train))
        rf_test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_rf_test))
        rf_train_r2 = r2_score(y_train, y_pred_rf_train)
        rf_test_r2 = r2_score(y_test, y_pred_rf_test)
        rf_train_mape = mean_absolute_percentage_error(y_train, y_pred_rf_train) * 100
        rf_test_mape = mean_absolute_percentage_error(y_test, y_pred_rf_test) * 100
        
        print("✓ Random Forest trained!")
        
        # ===== COMPARISON RESULTS =====
        print(f"\n{'=' * 70}")
        print("MODEL COMPARISON - LINEAR REGRESSION vs RANDOM FOREST")
        print(f"{'=' * 70}\n")
        
        print(f"{'METRIC':<30} {'LINEAR REGRESSION':<25} {'RANDOM FOREST':<25}")
        print("-" * 80)
        print(f"{'Train RMSE (€)':<30} {lr_train_rmse:>24.2f} {rf_train_rmse:>24.2f}")
        print(f"{'Test RMSE (€)':<30} {lr_test_rmse:>24.2f} {rf_test_rmse:>24.2f}")
        print(f"{'Train R² Score':<30} {lr_train_r2:>24.4f} {rf_train_r2:>24.4f}")
        print(f"{'Test R² Score':<30} {lr_test_r2:>24.4f} {rf_test_r2:>24.4f}")
        print(f"{'Train MAPE (%)':<30} {lr_train_mape:>24.2f} {rf_train_mape:>24.2f}")
        print(f"{'Test MAPE (%)':<30} {lr_test_mape:>24.2f} {rf_test_mape:>24.2f}")
        print("-" * 80)
        
        # Determine winner
        lr_score = lr_test_r2
        rf_score = rf_test_r2
        
        if rf_score > lr_score:
            print(f"\n🏆 WINNER: Random Forest (R² = {rf_score:.4f} vs {lr_score:.4f})")
        else:
            print(f"\n🏆 WINNER: Linear Regression (R² = {lr_score:.4f} vs {rf_score:.4f})")
        
        # Store results
        self.results = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred_lr_train': y_pred_lr_train,
            'y_pred_lr_test': y_pred_lr_test,
            'y_pred_rf_train': y_pred_rf_train,
            'y_pred_rf_test': y_pred_rf_test,
            'lr_metrics': {
                'train_rmse': lr_train_rmse,
                'test_rmse': lr_test_rmse,
                'train_r2': lr_train_r2,
                'test_r2': lr_test_r2,
                'train_mape': lr_train_mape,
                'test_mape': lr_test_mape
            },
            'rf_metrics': {
                'train_rmse': rf_train_rmse,
                'test_rmse': rf_test_rmse,
                'train_r2': rf_train_r2,
                'test_r2': rf_test_r2,
                'train_mape': rf_train_mape,
                'test_mape': rf_test_mape
            },
            'df': df,
            'feature_cols': feature_cols
        }
        
        return self.model_lr, self.model_rf
    
    def predict_future(self, days=10):
        """Predict future prices using both models"""
        if self.model_lr is None or self.model_rf is None:
            raise ValueError("Models not trained yet. Call train_models() first.")
        
        print(f"\n{'=' * 70}")
        print(f"FUTURE PREDICTIONS - NEXT {days} DAYS")
        print(f"{'=' * 70}\n")
        
        df = self.results['df']
        feature_cols = self.results['feature_cols']
        
        last_features = df[feature_cols].iloc[-1].values
        
        predictions_lr = []
        predictions_rf = []
        current_features = last_features.reshape(1, -1)
        
        print(f"{'Day':<6} {'Date':<12} {'Linear Reg (€)':<18} {'Random Forest (€)':<18} {'Difference':<12}")
        print("-" * 70)
        
        for day in range(1, days + 1):
            # Scale features
            scaled_features = self.scaler.transform(current_features)
            
            # Make predictions
            pred_lr = self.model_lr.predict(scaled_features)[0]
            pred_rf = self.model_rf.predict(scaled_features)[0]
            
            predictions_lr.append(pred_lr)
            predictions_rf.append(pred_rf)
            
            # Update Close price for next prediction
            avg_pred = (pred_lr + pred_rf) / 2
            current_features[0, 3] = avg_pred
            
            # Predict date
            last_date = df.index[-1]
            pred_date = last_date + timedelta(days=day)
            
            diff = pred_rf - pred_lr
            
            print(f"{day:<6} {str(pred_date.date()):<12} €{pred_lr:>15.2f} €{pred_rf:>16.2f} €{diff:>10.2f}")
        
        return predictions_lr, predictions_rf
    
    def plot_comparison(self):
        """Visualize model comparison"""
        if self.results is None:
            raise ValueError("No results to plot. Train models first.")
        
        y_test = self.results['y_test']
        y_pred_lr = self.results['y_pred_lr_test']
        y_pred_rf = self.results['y_pred_rf_test']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'Model Comparison: Linear Regression vs Random Forest - {self.ticker} (10 Years Data)', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Actual vs LR Predictions
        axes[0, 0].plot(y_test, label='Actual Price', color='blue', linewidth=2)
        axes[0, 0].plot(y_pred_lr, label='LR Predictions', color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_title('Linear Regression: Actual vs Predicted')
        axes[0, 0].set_xlabel('Time Steps')
        axes[0, 0].set_ylabel('Price (€)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Actual vs RF Predictions
        axes[0, 1].plot(y_test, label='Actual Price', color='blue', linewidth=2)
        axes[0, 1].plot(y_pred_rf, label='RF Predictions', color='green', linestyle='--', linewidth=2)
        axes[0, 1].set_title('Random Forest: Actual vs Predicted')
        axes[0, 1].set_xlabel('Time Steps')
        axes[0, 1].set_ylabel('Price (€)')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Both Predictions Overlay
        axes[0, 2].plot(y_test, label='Actual', color='blue', linewidth=2.5)
        axes[0, 2].plot(y_pred_lr, label='Linear Regression', color='red', linestyle='--', linewidth=1.5, alpha=0.8)
        axes[0, 2].plot(y_pred_rf, label='Random Forest', color='green', linestyle='--', linewidth=1.5, alpha=0.8)
        axes[0, 2].set_title('Model Predictions Comparison')
        axes[0, 2].set_xlabel('Time Steps')
        axes[0, 2].set_ylabel('Price (€)')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: LR Error
        error_lr = y_test - y_pred_lr
        axes[1, 0].plot(error_lr, color='red', linewidth=1.5, label='LR Error')
        axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        axes[1, 0].set_title('Linear Regression: Prediction Error')
        axes[1, 0].set_xlabel('Time Steps')
        axes[1, 0].set_ylabel('Error (€)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: RF Error
        error_rf = y_test - y_pred_rf
        axes[1, 1].plot(error_rf, color='green', linewidth=1.5, label='RF Error')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        axes[1, 1].set_title('Random Forest: Prediction Error')
        axes[1, 1].set_xlabel('Time Steps')
        axes[1, 1].set_ylabel('Error (€)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Scatter Comparison
        axes[1, 2].scatter(y_test, y_pred_lr, alpha=0.5, color='red', label='Linear Regression', s=20)
        axes[1, 2].scatter(y_test, y_pred_rf, alpha=0.5, color='green', label='Random Forest', s=20)
        axes[1, 2].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
                        'b--', linewidth=2, label='Perfect Prediction')
        axes[1, 2].set_title('Actual vs Predicted (Scatter)')
        axes[1, 2].set_xlabel('Actual Price (€)')
        axes[1, 2].set_ylabel('Predicted Price (€)')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        print("✓ Visualization complete!")


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("DUAL MODEL STOCK PREDICTION - 10 YEARS DATA")
    print("=" * 70)
    
    # Select Siemens for 10 years of analysis
    ticker = 'SIE.DE'  # Siemens AG
    stock_name = 'Siemens'
    
    print(f"\n📊 Analyzing: {stock_name} ({ticker})")
    
    try:
        # Create predictor
        predictor = DualModelStockPredictor(ticker=ticker, period='10y')
        
        # Fetch data
        predictor.fetch_data()
        
        # Train both models
        predictor.train_models()
        
        # Get current stock info
        current_price = predictor.data['Close'].iloc[-1]
        print(f"\n{'=' * 70}")
        print("CURRENT STOCK INFORMATION")
        print(f"{'=' * 70}")
        print(f"Current Price: €{current_price:.2f}")
        print(f"52-Week High: €{predictor.data['Close'].tail(252).max():.2f}")
        print(f"52-Week Low: €{predictor.data['Close'].tail(252).min():.2f}")
        print(f"10-Year High: €{predictor.data['Close'].max():.2f}")
        print(f"10-Year Low: €{predictor.data['Close'].min():.2f}")
        
        # Make predictions
        pred_lr, pred_rf = predictor.predict_future(days=10)
        
        # Analysis
        print(f"\n{'=' * 70}")
        print("PREDICTION ANALYSIS")
        print(f"{'=' * 70}")
        
        next_day_lr = pred_lr[0]
        next_day_rf = pred_rf[0]
        next_day_avg = (next_day_lr + next_day_rf) / 2
        
        change_lr = next_day_lr - current_price
        change_rf = next_day_rf - current_price
        change_pct_lr = (change_lr / current_price) * 100
        change_pct_rf = (change_rf / current_price) * 100
        
        print(f"\nLinear Regression Next Day Prediction:")
        print(f"  Price: €{next_day_lr:.2f}")
        print(f"  Change: {change_pct_lr:+.2f}% (€{change_lr:+.2f})")
        
        print(f"\nRandom Forest Next Day Prediction:")
        print(f"  Price: €{next_day_rf:.2f}")
        print(f"  Change: {change_pct_rf:+.2f}% (€{change_rf:+.2f})")
        
        print(f"\nConsensus (Average):")
        print(f"  Price: €{next_day_avg:.2f}")
        print(f"  Change: {((next_day_avg - current_price) / current_price * 100):+.2f}%")
        
        # Trend analysis
        if next_day_avg > current_price:
            print(f"\n📈 Overall Trend: BULLISH (Upward)")
        else:
            print(f"\n📉 Overall Trend: BEARISH (Downward)")
        
        # Visualize
        predictor.plot_comparison()
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nRequired packages: yfinance pandas numpy scikit-learn matplotlib")
        print("Install with: pip install yfinance pandas numpy scikit-learn matplotlib")
