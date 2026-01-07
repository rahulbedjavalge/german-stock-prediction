import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.datasets import make_classification

# ============================================
# LOGISTIC REGRESSION EXAMPLE
# ============================================

# Generate sample dataset
X, y = make_classification(
    n_samples=200,      # 200 data points
    n_features=2,       # 2 features for visualization
    n_informative=2,    # Both features are informative
    n_redundant=0,      # No redundant features
    random_state=42,
    n_clusters_per_class=1
)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the logistic regression model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("=" * 50)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 50)
print(f"\nAccuracy: {accuracy:.2%}")
print(f"\nConfusion Matrix:\n{conf_matrix}")
print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

# Display model coefficients and intercept
print(f"\nModel Coefficients: {model.coef_}")
print(f"Model Intercept: {model.intercept_}")

# Visualization
plt.figure(figsize=(10, 5))

# Plot 1: Training data with decision boundary
plt.subplot(1, 2, 1)
plt.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], 
            label='Class 0', alpha=0.6, color='blue')
plt.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], 
            label='Class 1', alpha=0.6, color='red')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Training Data')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Predictions vs Actual
plt.subplot(1, 2, 2)
plt.scatter(range(len(y_test)), y_test, label='Actual', alpha=0.6, color='blue')
plt.scatter(range(len(y_pred)), y_pred, label='Predicted', alpha=0.6, color='red')
plt.xlabel('Test Sample Index')
plt.ylabel('Class')
plt.title('Predictions vs Actual')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
