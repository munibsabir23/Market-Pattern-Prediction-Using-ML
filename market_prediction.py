# Market Pattern Prediction using Machine Learning
# Author: Qazi Munib Sabir

import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 1: Download historical stock data
data = yf.download("AAPL", start="2020-01-01", end="2025-01-01")
data.dropna(inplace=True)

# Step 2: Feature engineering - technical indicators
data["SMA_10"] = data["Close"].rolling(window=10).mean()
data["SMA_30"] = data["Close"].rolling(window=30).mean()
data["EMA_10"] = data["Close"].ewm(span=10, adjust=False).mean()
data["Return"] = data["Close"].pct_change()
data["Volatility"] = data["Return"].rolling(window=10).std()
data["RSI"] = 100 - (100 / (1 + (data["Return"].rolling(14).mean() / data["Return"].rolling(14).std())))
data.dropna(inplace=True)

# Step 3: Define target variable (1 if price goes up next day, else 0)
data["Target"] = np.where(data["Close"].shift(-1) > data["Close"], 1, 0)

# Step 4: Prepare features and target
X = data[["SMA_10", "SMA_30", "EMA_10", "Return", "Volatility", "RSI"]]
y = data["Target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=False)

# Scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 5: Train models
log_reg = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=200, random_state=42)

log_reg.fit(X_train_scaled, y_train)
rf.fit(X_train, y_train)

# Step 6: Evaluate
y_pred_lr = log_reg.predict(X_test_scaled)
y_pred_rf = rf.predict(X_test)

acc_lr = accuracy_score(y_test, y_pred_lr)
acc_rf = accuracy_score(y_test, y_pred_rf)

print("=== Logistic Regression ===")
print(f"Accuracy: {acc_lr:.2f}")
print(classification_report(y_test, y_pred_lr))

print("=== Random Forest ===")
print(f"Accuracy: {acc_rf:.2f}")
print(classification_report(y_test, y_pred_rf))

# Step 7: Visualize feature importance
importances = pd.Series(rf.feature_importances_, index=X.columns)
importances.sort_values().plot(kind='barh', figsize=(8, 5), title="Feature Importance")
plt.show()
