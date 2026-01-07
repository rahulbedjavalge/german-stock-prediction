import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.datasets import make_regression

# ============================================
# LINEAR REGRESSION EXAMPLE
# ============================================

# Generate sample dataset
X, y = make_regression(
    n_samples=100,       # 100 data points
    n_features=1,        # 1 feature for visualization
    noise=20,            # Add some noise to data
    random_state=42
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Evaluate the model
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print("=" * 50)
print("LINEAR REGRESSION RESULTS")
print("=" * 50)
print(f"\nModel Coefficient (Slope): {model.coef_[0]:.4f}")
print(f"Model Intercept: {model.intercept_:.4f}")
print(f"\nTraining Metrics:")
print(f"  MSE (Mean Squared Error): {train_mse:.4f}")
print(f"  RMSE (Root Mean Squared Error): {train_rmse:.4f}")
print(f"  MAE (Mean Absolute Error): {train_mae:.4f}")
print(f"  R² Score: {r2_train:.4f}")
print(f"\nTesting Metrics:")
print(f"  MSE: {test_mse:.4f}")
print(f"  RMSE: {test_rmse:.4f}")
print(f"  MAE: {test_mae:.4f}")
print(f"  R² Score: {r2_test:.4f}")

# Equation of the line: y = mx + b
print(f"\nLinear Equation: y = {model.coef_[0]:.4f}x + {model.intercept_:.4f}")

# Visualization
plt.figure(figsize=(14, 5))

# Plot 1: Training data with fitted line
plt.subplot(1, 3, 1)
plt.scatter(X_train, y_train, label='Training Data', alpha=0.6, color='blue')
plt.plot(X_train, y_pred_train, 'r-', linewidth=2, label='Fitted Line')
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('Training Data with Fitted Line')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Testing data with predictions
plt.subplot(1, 3, 2)
plt.scatter(X_test, y_test, label='Test Data', alpha=0.6, color='green')
plt.plot(X_test, y_pred_test, 'r-', linewidth=2, label='Predictions')
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('Test Data with Predictions')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Residuals (Errors)
plt.subplot(1, 3, 3)
residuals = y_test - y_pred_test
plt.scatter(y_pred_test, residuals, alpha=0.6, color='purple')
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional: Predict for new data
print(f"\n--- Predicting for New Data ---")
new_X = np.array([[1.5], [3.0], [-1.0]])
new_predictions = model.predict(new_X)
for x_val, y_val in zip(new_X, new_predictions):
    print(f"X = {x_val[0]:.2f} → Predicted y = {y_val:.4f}")
