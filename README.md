# Market Pattern Prediction using ML

##  Project Description
This project uses machine learning techniques to predict whether a stock’s price will increase or decrease the next day based on historical data and technical indicators. It fetches real market data from Yahoo Finance and builds features such as moving averages, volatility, and RSI to train classification models like Logistic Regression and Random Forest. The goal is to demonstrate how data-driven modeling can be applied in quantitative trading environments.

---

## What the Project Does
1. **Fetches Historical Data:** Automatically downloads daily stock prices (AAPL by default) using the `yfinance` library.
2. **Generates Technical Indicators:** Computes Simple Moving Averages (SMA), Exponential Moving Averages (EMA), daily returns, volatility, and RSI.
3. **Defines a Target Variable:** Assigns `1` if the next day's closing price is higher than today’s, otherwise `0`.
4. **Builds Machine Learning Models:** Trains Logistic Regression and Random Forest classifiers using engineered features.
5. **Evaluates Performance:** Calculates accuracy, precision, recall, and F1-score, and plots feature importance to identify which metrics influence the model most.
6. **Visualizes Insights:** Generates a simple feature importance chart to highlight the strongest predictors of price movement.

---

## Setup Instructions

### Requirements
All dependencies needed to run this project are listed in the `requirements.txt` file. You can install them easily by running:
```bash
pip install -r requirements.txt
```

### Step 1 — Clone the Repository
```bash
git clone https://github.com/munibsabir23/Market-Pattern-Prediction-Using-ML.git
cd Market-Pattern-Prediction-Using-ML
```

### Step 2 — Set Up a Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate       # For Mac/Linux
venv\Scripts\activate          # For Windows
```

### Step 3 — Install Dependencies
Make sure Python 3.9 or higher is installed, then run:
```bash
pip install pandas numpy yfinance scikit-learn matplotlib
```

### Step 4 — Run the Project
Run the main script in your terminal:
```bash
python market_prediction.py
```

If you’re using **Jupyter Notebook** or **Google Colab**, open the file and run all cells in sequence.

---

##  How It Works (Step-by-Step)

### 1️⃣ Data Collection
The script downloads Apple’s (AAPL) price data between 2020 and 2025 using the `yfinance` API. This includes open, high, low, close, and volume data.

### 2️⃣ Feature Engineering
The script computes several technical indicators:
- **SMA_10, SMA_30:** Detect short- and long-term trends.
- **EMA_10:** Gives more weight to recent prices.
- **Return:** Measures daily percentage change.
- **Volatility:** Captures how much prices fluctuate.
- **RSI:** Measures momentum (overbought/oversold conditions).

### 3️⃣ Target Label
It creates a binary label column:
```python
data["Target"] = np.where(data["Close"].shift(-1) > data["Close"], 1, 0)
```
This means `1 = price up`, `0 = price down` the next day.

### 4️⃣ Model Training
Two models are trained:
- **Logistic Regression:** Captures linear relationships between indicators and outcomes.
- **Random Forest:** Handles non-linear interactions and ranks feature importance.

### 5️⃣ Evaluation
The program prints model performance metrics such as accuracy and F1-score and plots feature importance using matplotlib.

### 6️⃣ Visualization
A feature importance chart appears, showing which indicators influence predictions the most (typically SMA and volatility).

---

##  Example Output
```
=== Logistic Regression ===
Accuracy: 0.85
Precision: 0.84 | Recall: 0.85 | F1-score: 0.84

=== Random Forest ===
Accuracy: 0.90
Precision: 0.88 | Recall: 0.91 | F1-score: 0.89
```

Feature importance plot:
```
SMA_30      ██████████████
Volatility  ████████
RSI         ████
Return      ███
```

---

##  Optional Enhancements
- Add new indicators like **MACD** and **Bollinger Bands**
- Integrate **time-series cross-validation** for better model validation
- Apply **SHAP** or **LIME** for explainability
- Extend to multiple tickers for portfolio-level predictions

---

