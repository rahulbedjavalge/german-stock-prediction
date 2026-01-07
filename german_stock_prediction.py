import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================
# GERMAN STOCK MARKET PRICE PREDICTION MODEL
# ============================================

class GermanStockPredictor:
    """
    A machine learning model to predict German stock prices
    """
    
    def __init__(self, ticker, period='2y'):
        """
        Initialize the predictor
        
        Parameters:
        - ticker: Stock ticker symbol (e.g., 'SAP.DE', 'VOW3.DE', 'SIE.DE')
        - period: Historical data period (default: 2 years)
        """
        self.ticker = ticker
        self.period = period
        self.data = None
        self.model = None
        self.scaler = MinMaxScaler()
        
    def fetch_data(self):
        """Fetch stock data from Yahoo Finance"""
        print(f"Fetching data for {self.ticker}...")
        stock = yf.Ticker(self.ticker)
        self.data = stock.history(period=self.period)
        
        if self.data.empty:
            raise ValueError(f"No data found for ticker {self.ticker}")
        
        print(f"Successfully fetched {len(self.data)} days of data")
        print(f"Date range: {self.data.index[0].date()} to {self.data.index[-1].date()}")
        return self.data
    
    def create_features(self, data):
        """Create technical indicators and features for prediction"""
        df = data.copy()
        
        # Moving Averages
        df['MA_5'] = df['Close'].rolling(window=5).mean()
        df['MA_10'] = df['Close'].rolling(window=10).mean()
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_50'] = df['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        
        # MACD (Moving Average Convergence Divergence)
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        
        # RSI (Relative Strength Index)
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
        
        # Price Rate of Change
        df['ROC'] = ((df['Close'] - df['Close'].shift(10)) / df['Close'].shift(10)) * 100
        
        # Volume changes
        df['Volume_Change'] = df['Volume'].pct_change()
        
        # Price changes
        df['Price_Change'] = df['Close'].pct_change()
        df['High_Low_Diff'] = df['High'] - df['Low']
        df['Open_Close_Diff'] = df['Close'] - df['Open']
        
        # Lag features (previous day prices)
        for i in range(1, 6):
            df[f'Close_Lag_{i}'] = df['Close'].shift(i)
        
        # Target: Next day's closing price
        df['Target'] = df['Close'].shift(-1)
        
        # Drop rows with NaN values
        df = df.dropna()
        
        # Replace any infinity values with NaN and drop them
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        
        return df
    
    def prepare_data(self):
        """Prepare data for training"""
        print("\nCreating features and preparing data...")
        df = self.create_features(self.data)
        
        # Select features for training
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA_5', 'MA_10', 'MA_20', 'MA_50',
            'EMA_12', 'EMA_26', 'MACD', 'RSI',
            'BB_middle', 'BB_upper', 'BB_lower',
            'ROC', 'Volume_Change', 'Price_Change',
            'High_Low_Diff', 'Open_Close_Diff',
            'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3',
            'Close_Lag_4', 'Close_Lag_5'
        ]
        
        X = df[feature_columns].values
        y = df['Target'].values
        
        # Split data (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False  # Don't shuffle time series data
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, df
    
    def train_model(self, model_type='random_forest'):
        """Train the prediction model"""
        X_train, X_test, y_train, y_test, df = self.prepare_data()
        
        print(f"\nTraining {model_type} model...")
        
        if model_type == 'linear_regression':
            self.model = LinearRegression()
        elif model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            raise ValueError("Model type must be 'linear_regression' or 'random_forest'")
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Make predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Evaluate model
        train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        train_mae = mean_absolute_error(y_train, y_pred_train)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        train_mape = np.mean(np.abs((y_train - y_pred_train) / y_train)) * 100
        test_mape = np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100
        
        print("\n" + "=" * 60)
        print(f"MODEL PERFORMANCE - {self.ticker}")
        print("=" * 60)
        print(f"\nTraining Metrics:")
        print(f"  RMSE: €{train_rmse:.2f}")
        print(f"  MAE:  €{train_mae:.2f}")
        print(f"  MAPE: {train_mape:.2f}%")
        print(f"  R² Score: {train_r2:.4f}")
        
        print(f"\nTesting Metrics:")
        print(f"  RMSE: €{test_rmse:.2f}")
        print(f"  MAE:  €{test_mae:.2f}")
        print(f"  MAPE: {test_mape:.2f}%")
        print(f"  R² Score: {test_r2:.4f}")
        
        # Store results for plotting
        self.results = {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'y_pred_train': y_pred_train,
            'y_pred_test': y_pred_test,
            'df': df
        }
        
        return self.model
    
    def predict_future(self, days=5):
        """Predict future stock prices"""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        print(f"\n{'=' * 60}")
        print(f"FUTURE PREDICTIONS FOR {self.ticker}")
        print(f"{'=' * 60}")
        
        # Get the last row of features
        df = self.results['df']
        
        # Use the same features we used for training
        feature_columns = [
            'Open', 'High', 'Low', 'Close', 'Volume',
            'MA_5', 'MA_10', 'MA_20', 'MA_50',
            'EMA_12', 'EMA_26', 'MACD', 'RSI',
            'BB_middle', 'BB_upper', 'BB_lower',
            'ROC', 'Volume_Change', 'Price_Change',
            'High_Low_Diff', 'Open_Close_Diff',
            'Close_Lag_1', 'Close_Lag_2', 'Close_Lag_3',
            'Close_Lag_4', 'Close_Lag_5'
        ]
        
        last_features = df[feature_columns].iloc[-1].values
        
        predictions = []
        current_features = last_features.reshape(1, -1)
        
        for day in range(1, days + 1):
            # Scale features
            scaled_features = self.scaler.transform(current_features)
            
            # Make prediction
            pred_price = self.model.predict(scaled_features)[0]
            predictions.append(pred_price)
            
            # Update features for next prediction (simple approach)
            # In a more sophisticated model, you'd update all features
            current_features[0, 3] = pred_price  # Update Close price
            
            # Predict date
            last_date = df.index[-1]
            pred_date = last_date + timedelta(days=day)
            
            print(f"Day {day} ({pred_date.date()}): €{pred_price:.2f}")
        
        return predictions
    
    def plot_results(self):
        """Visualize predictions and actual prices"""
        if self.results is None:
            raise ValueError("No results to plot. Train model first.")
        
        y_test = self.results['y_test']
        y_pred_test = self.results['y_pred_test']
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        fig.suptitle(f'Stock Price Prediction Results - {self.ticker}', 
                     fontsize=16, fontweight='bold')
        
        # Plot 1: Actual vs Predicted (Test Set)
        axes[0, 0].plot(y_test, label='Actual Price', color='blue', linewidth=2)
        axes[0, 0].plot(y_pred_test, label='Predicted Price', 
                        color='red', linestyle='--', linewidth=2)
        axes[0, 0].set_title('Actual vs Predicted Prices (Test Set)')
        axes[0, 0].set_xlabel('Time Steps')
        axes[0, 0].set_ylabel('Price (€)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Prediction Error
        error = y_test - y_pred_test
        axes[0, 1].plot(error, color='purple', linewidth=1.5)
        axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        axes[0, 1].set_title('Prediction Error Over Time')
        axes[0, 1].set_xlabel('Time Steps')
        axes[0, 1].set_ylabel('Error (€)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Scatter Plot (Actual vs Predicted)
        axes[1, 0].scatter(y_test, y_pred_test, alpha=0.6, color='green')
        axes[1, 0].plot([y_test.min(), y_test.max()], 
                        [y_test.min(), y_test.max()], 
                        'r--', linewidth=2, label='Perfect Prediction')
        axes[1, 0].set_title('Actual vs Predicted Scatter Plot')
        axes[1, 0].set_xlabel('Actual Price (€)')
        axes[1, 0].set_ylabel('Predicted Price (€)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Historical Price with Moving Averages
        df = self.results['df']
        last_200 = df.tail(200)
        axes[1, 1].plot(last_200.index, last_200['Close'], 
                        label='Close Price', color='blue', linewidth=2)
        axes[1, 1].plot(last_200.index, last_200['MA_20'], 
                        label='MA 20', color='orange', linestyle='--', linewidth=1.5)
        axes[1, 1].plot(last_200.index, last_200['MA_50'], 
                        label='MA 50', color='green', linestyle='--', linewidth=1.5)
        axes[1, 1].set_title('Historical Price with Moving Averages')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Price (€)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        print("\n✓ Visualization complete!")


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Popular German stocks (DAX components)
    german_stocks = {
        'SAP': 'SAP.DE',           # SAP SE
        'Siemens': 'SIE.DE',       # Siemens AG
        'Volkswagen': 'VOW3.DE',   # Volkswagen
        'Allianz': 'ALV.DE',       # Allianz SE
        'Deutsche Bank': 'DBK.DE', # Deutsche Bank
        'BMW': 'BMW.DE',           # BMW
        'Adidas': 'ADS.DE',        # Adidas
        'BASF': 'BAS.DE',          # BASF SE
        'Bayer': 'BAYN.DE',        # Bayer AG
        'Mercedes': 'MBG.DE',      # Mercedes-Benz Group
    }
    
    print("=" * 60)
    print("GERMAN STOCK MARKET PRICE PREDICTION")
    print("=" * 60)
    print("\nAvailable German stocks:")
    for i, (name, ticker) in enumerate(german_stocks.items(), 1):
        print(f"{i}. {name} ({ticker})")
    
    # Default: Predict SAP stock
    selected_ticker = 'SAP.DE'
    selected_name = 'SAP'
    
    print(f"\n{'=' * 60}")
    print(f"Analyzing {selected_name} ({selected_ticker})")
    print(f"{'=' * 60}")
    
    try:
        # Create predictor instance
        predictor = GermanStockPredictor(ticker=selected_ticker, period='2y')
        
        # Fetch stock data
        predictor.fetch_data()
        
        # Train the model (using Random Forest for better accuracy)
        predictor.train_model(model_type='random_forest')
        
        # Predict next 5 days
        future_predictions = predictor.predict_future(days=5)
        
        # Visualize results
        predictor.plot_results()
        
        # Show current stock info
        print(f"\n{'=' * 60}")
        print("CURRENT STOCK INFORMATION")
        print(f"{'=' * 60}")
        current_price = predictor.data['Close'].iloc[-1]
        print(f"Current Price: €{current_price:.2f}")
        print(f"Day's High: €{predictor.data['High'].iloc[-1]:.2f}")
        print(f"Day's Low: €{predictor.data['Low'].iloc[-1]:.2f}")
        print(f"Volume: {predictor.data['Volume'].iloc[-1]:,.0f}")
        
        # Price change analysis
        price_change = future_predictions[0] - current_price
        price_change_pct = (price_change / current_price) * 100
        
        print(f"\nNext Day Prediction:")
        print(f"  Predicted Price: €{future_predictions[0]:.2f}")
        print(f"  Expected Change: €{price_change:+.2f} ({price_change_pct:+.2f}%)")
        
        if price_change > 0:
            print(f"  📈 Trend: BULLISH (Upward)")
        else:
            print(f"  📉 Trend: BEARISH (Downward)")
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        print("\nNote: Make sure you have internet connection and the required packages:")
        print("  pip install yfinance pandas numpy scikit-learn matplotlib")
