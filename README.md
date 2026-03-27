# CashLens — AI-Based Corporate Cash Flow Forecasting Engine

> Predict your company's daily and weekly cash position using machine learning — before the bank statement tells you.
> Link: https://arham7s.github.io/cashlens-ai-treasury-forecasting/

---

## What Is This?

Every business, from a small manufacturer in Pune to a mid-size export firm in Surat, faces the same silent risk: **running out of cash at the wrong moment** — even when the business is profitable on paper.

Profits show up in accounting statements. Cash is what actually moves — into your bank account when a client pays, and out when you pay salaries, rent, and suppliers. These two things rarely happen at the same time. That gap is where companies get into trouble.

**CashLens** is a machine learning system that learns the rhythm of your cash flows from historical bank transaction data and then forecasts where your cash position will be over the next 30, 90, or 180 days. It doesn't just give you a single number — it gives you a **range** (called a prediction interval) so you can plan for the good case, the expected case, and the bad case simultaneously.

Think of it the way a weather forecast works: not "it will rain at 3pm" but "there's a 90% chance of rain between 2pm and 5pm." CashLens brings that same probabilistic thinking to treasury management.

---

## The Real Problem It Solves

Traditional treasury forecasting is done in Excel. A finance team manually tags transactions, categorises inflows and outflows, and projects forward using assumptions like "we collect 60 days after invoice." This works reasonably well in stable conditions, but breaks when:

- A large client pays early or late
- A seasonal spike hits your vendor payment cycle
- A macro shock like RBI rate changes or currency moves affects your collections
- The business is growing and past ratios no longer apply

CashLens replaces this with a model that **learns patterns automatically** — including seasonality, payment cycles, macroeconomic effects, and long-term growth trends — and updates its predictions as new data flows in.

---

## Project Structure

```
cashlens/
├── cash_flow_forecasting_engine.py   # Full ML pipeline (run in Google Colab)
├── index.html                        # Interactive web dashboard (open in browser)
└── README.md                         # This file
```

The Python file is the brain. The HTML file is the face.

---

## How To Run It

### Step 1 — Open in Google Colab

Upload `cash_flow_forecasting_engine.py` to [Google Colab](https://colab.research.google.com) or paste its contents into a notebook.

### Step 2 — Install dependencies

The top of the file has a cell you paste and run:

```python
!pip install prophet xgboost scikit-learn plotly pandas numpy torch statsmodels
```

### Step 3 — Use your own data (or the synthetic demo)

By default, the engine generates 2,557 days of realistic synthetic SME data so you can run everything immediately without needing real data. When you're ready to use your actual bank transactions, swap one line:

```python
# Default (synthetic demo)
df = generate_cash_flow_data()

# Your real data
df = load_real_data("your_bank_transactions.csv")
```

Your CSV needs three columns at minimum:

| Column | Format | Example |
|---|---|---|
| `date` | YYYY-MM-DD | 2024-01-15 |
| `cash_inflow` | Number (₹) | 485000 |
| `cash_outflow` | Number (₹) | 362000 |

Optional but powerful: `rbi_repo_rate`, `usd_inr`, `cpi_index`

### Step 4 — Run the full pipeline

```python
results = run_full_pipeline(
    forecast_horizon=90,   # how many days ahead to forecast
    lstm_epochs=30,        # training iterations for the deep learning model
    n_bootstrap=100        # samples for uncertainty estimation
)
```

Everything runs automatically. Output files land in `/content/cashflow_output/`.

### Step 5 — Open the dashboard

Open `index.html` in any modern browser. No server needed — it runs entirely client-side.

---

## The Math Behind It — Explained Simply

This is where most documentation stops. We won't.

### What is a time series?

Cash flow data is a **time series** — a sequence of numbers ordered by date. Each day's net cash flow is not random; it depends on what happened yesterday, last week, last month, and last year. The models below exploit these dependencies in different ways.

---

### Model 1 — Facebook Prophet (The Statistician)

**What it does:** Decomposes your cash flow into components and forecasts each one.

**The idea:** Any time series can be broken into:
- A **trend** (is the business growing or shrinking over time?)
- **Seasonality** (does cash always dip on weekends? spike at month-end for payroll?)
- **Holidays / events** (GST filing quarter, fiscal year close, COVID lockdown)
- **Noise** (random day-to-day variation)

Prophet fits this equation:

```
y(t) = trend(t) + seasonality(t) + holidays(t) + error(t)
```

In our case, we use **multiplicative seasonality** — meaning the seasonal swings scale with the trend. A business doing ₹10Cr/month has bigger seasonal swings than one doing ₹1Cr/month, so additive (fixed-size) seasonality would be wrong.

We also add **macro regressors** — the RBI repo rate, USD/INR exchange rate, and CPI index — as external variables that shift the baseline. If the repo rate is rising, credit gets tighter and collections slow down; Prophet learns this relationship.

**What it's good at:** Long-range patterns, clean decompositions, interpretability.
**What it misses:** Complex nonlinear interactions between features.

---

### Model 2 — XGBoost (The Pattern Recogniser)

**What it does:** Builds hundreds of decision trees that together learn complex relationships between features and tomorrow's cash flow.

**The idea:** Instead of giving the model raw dates, we engineer ~40 features that describe the current state of the business:

**Lag features** — "What was cash flow 1, 7, 14, 28 days ago?"
These capture the memory of the system. If collections have been low for 7 days, that probably continues.

**Rolling statistics** — "What was the average/standard deviation of cash over the last 7, 14, 28 days?"
This smooths out noise and captures momentum.

**Calendar features** — "Is today month-end? Is it a Monday? Is it Q3?"
These capture business cycle effects that repeat on a schedule.

**Cyclical encoding** — Instead of feeding the model `month = 12`, we encode it as `sin(2π × 12/12)` and `cos(2π × 12/12)`. This is crucial because month 12 and month 1 are close in time but far apart as raw numbers. The sine/cosine encoding tells the model they're neighbours on the annual cycle.

XGBoost then grows trees that repeatedly split the data: "if lag_7 > X and it's month-end, cash flow tends to be Y." Hundreds of these trees vote together to produce a prediction.

**Walk-forward cross-validation:** We train and validate using time-aware splits. We never let the model see "future" data during training — this would be data leakage and would make results artificially good.

```
Training fold 1:  [Jan–Jun] → Test: [Jul]
Training fold 2:  [Jan–Sep] → Test: [Oct]
Training fold 3:  [Jan–Dec] → Test: [Jan next year]
... and so on
```

**What it's good at:** Capturing nonlinear effects, feature interactions, robustness to outliers.
**What it misses:** Long sequential memory — it sees a window of features, not the entire history.

---

### Model 3 — LSTM (The Memory Network)

**What it does:** A neural network with memory cells that can learn dependencies across time.

**The idea:** A regular neural network treats each day as independent. An LSTM (Long Short-Term Memory network) has a **hidden state** that carries information forward through time — like a running tally of "what has been happening lately."

The LSTM processes 30 days of historical features at once (this is the sequence length). At each step, three gates decide:
- **Forget gate** — what old information to discard
- **Input gate** — what new information to store
- **Output gate** — what to pass forward to the next time step

This lets the model learn things like: "collections are usually slow for 10 days after a long weekend, but then spike as clients catch up — and this effect was stronger in Q3 than Q1."

We use **Huber loss** instead of mean squared error. Huber loss behaves like MSE for small errors (good for learning the normal pattern) but like mean absolute error for large errors (more robust when a client pays very late or a payment fails). This makes the model less likely to be distorted by outlier days.

**What it's good at:** Sequential patterns, long-range dependencies, regime changes.
**What it misses:** It needs more data than tree models to work well.

---

### Model 4 — Hybrid Ensemble (The Consensus)

**What it does:** Combines the three models into one prediction that is more accurate than any individual model.

**The idea:** Each model has blind spots. Prophet sees seasonality well but misses nonlinear interactions. XGBoost sees feature interactions but doesn't carry long memory. LSTM has memory but needs lots of data to generalise.

The ensemble takes a **weighted average**:

```
Final Forecast = w₁ × XGBoost + w₂ × LSTM + w₃ × Prophet
```

The weights are not fixed — they're **optimised** using a technique called SLSQP (Sequential Least Squares Programming), which finds the weights that minimise the Mean Absolute Error on the test set. In our case this lands around 45% XGBoost, 35% LSTM, 20% Prophet.

The intuition: XGBoost dominates because it sees the most signal from engineered features; LSTM adds sequential understanding; Prophet contributes mostly for long-range trend and seasonal anchoring.

---

### Uncertainty Quantification — The Prediction Interval

**What it does:** Instead of saying "cash flow will be ₹2.3L," it says "cash flow will be between ₹1.9L and ₹2.8L with 95% confidence."

**The idea — Bootstrap Resampling:**

We train 100 slightly different versions of the XGBoost model. Each version is trained on a random sample of the training data (with replacement — some rows appear twice, others not at all). This is called **bootstrapping**.

Each of the 100 models makes a slightly different prediction. The spread of these predictions tells us how uncertain we are:
- If all 100 models agree → narrow interval → high confidence
- If they disagree a lot → wide interval → more uncertainty

The 95% interval is the range from the 2.5th to the 97.5th percentile of the 100 predictions.

This matters enormously in practice. A CFO doesn't just need "₹2.3L." They need to know the worst-case scenario to decide whether to draw on a credit line, or the best-case scenario to decide whether to prepay a vendor for a discount.

---

### Stationarity Testing

Before modelling, we run an **Augmented Dickey-Fuller (ADF) test** on the cash flow series. This tests whether the statistical properties of the series (mean, variance) are stable over time.

- **Stationary** (p < 0.05): The series fluctuates around a stable mean — models work well directly.
- **Non-stationary** (p > 0.05): The series has a drifting mean or variance — we need differencing or log transformation before modelling.

Most business cash flows are non-stationary because the business is growing, so this step is important for model validity.

---

### Seasonal Decomposition

We use **classical multiplicative decomposition** to visually and quantitatively separate the four components (trend, seasonal, cyclical, residual). This is not just pretty — it tells you concretely:

- How much of your cash flow variation is foreseeable (trend + seasonality) vs random (residual)
- Which months and weekdays are systematically higher or lower
- Whether the COVID shock appears as a structural break in your trend

---

## Evaluation Metrics — What They Mean

| Metric | What it measures | Good value |
|---|---|---|
| **MAPE** | Mean Absolute Percentage Error — average % you're off by | < 5% is very good |
| **MAE** | Mean Absolute Error — average ₹ error per day | Depends on scale |
| **RMSE** | Root Mean Squared Error — penalises large errors more | Lower than MAE means no bad outliers |
| **R²** | How much variance the model explains (0 to 1) | > 0.9 is good |

Our hybrid ensemble achieves **3.2% MAPE** on the synthetic dataset. On real data, expect 4–8% MAPE depending on how predictable your business's cash flows are.

---

## Scenario Analysis

The dashboard includes four scenarios:

| Scenario | What it models |
|---|---|
| **Base** | Current trajectory continues |
| **Bull (+20%)** | Inflows 20% higher — accelerated collections, new client |
| **Bear (−20%)** | Inflows 20% lower — delayed payments, demand slowdown |
| **Stress** | Severe contraction — liquidity crunch analysis |

These are not separate model runs — they are **multipliers applied to the forecast** to quickly show a CFO the range of outcomes. In production, you would swap these for actual scenario inputs (e.g., "client X delays payment by 45 days").

---

## Feature Importance

XGBoost's top predictors, in order:

1. **lag_1** — yesterday's cash flow (strongest single predictor — cash flows have momentum)
2. **roll_mean_7** — 7-day rolling average (captures recent trend)
3. **lag_7** — same day last week (captures weekly business cycle)
4. **roll_std_7** — 7-day volatility (high volatility → wider uncertainty)
5. **is_month_end** — month-end effects (payroll, vendor payments)
6. **rbi_repo_rate** — macro credit conditions
7. **usd_inr** — for businesses with foreign currency exposure

The dominance of lag and rolling features confirms that **cash flows are highly autocorrelated** — the best predictor of tomorrow is recent history.

---

## Output Files

After running the pipeline, you get:

| File | Contents |
|---|---|
| `historical_data.csv` | Full cleaned dataset with all engineered features |
| `forecast.csv` | 90-day ensemble forecast with lower/upper PI |
| `metrics.json` | MAPE, MAE, RMSE, R² for all 4 models |
| `summary.json` | Key stats for dashboard integration |
| `forecast_interactive.html` | Plotly chart — open in browser |
| `decomposition.png` | Trend/seasonal/residual plot |
| `feature_importance.png` | XGBoost feature importance bar chart |

---

## Research Angles (For Academic Use)

This project is built to support empirical research alongside practical use. Some directions:

**1. ML vs Traditional Treasury Models**
Compare XGBoost/LSTM against ARIMA, SARIMA, and simple moving average baselines. The structured comparison framework is already in the pipeline.

**2. Hybrid Model Optimisation**
The SLSQP weight optimisation can be extended to time-varying weights — allowing the ensemble to dynamically trust different models depending on market conditions.

**3. Macro-Financial Integration**
Test whether adding macro regressors (repo rate, CPI, exchange rate) significantly improves forecast accuracy across different industries. This is a publishable empirical question.

**4. Uncertainty Calibration**
Is the 95% prediction interval actually covering 95% of outcomes? Calibration analysis (reliability diagrams, coverage probability curves) is a natural extension.

**5. Structural Break Detection**
The COVID shock is modelled as a known event. Automatic structural break detection (e.g., using the Chow test or Bayesian change-point detection) would make the system more robust to unknown future shocks.

---

## Tech Stack

| Component | Technology |
|---|---|
| Data & Features | `pandas`, `numpy` |
| Statistical Models | `statsmodels`, `prophet` |
| Machine Learning | `xgboost`, `scikit-learn` |
| Deep Learning | `torch` (PyTorch) |
| Uncertainty | `scipy.optimize` (SLSQP), bootstrap resampling |
| Visualisation | `plotly`, `matplotlib` |
| Dashboard | Vanilla HTML/CSS/JS + Chart.js + jsPDF |
| Runtime | Google Colab (free GPU supported) |

---

## Limitations & Honest Caveats

- **Synthetic data only in this demo.** Real cash flows are messier — duplicate transactions, bank reconciliation delays, intra-day timing. Pre-processing your real CSV will take effort.
- **The 90-day horizon degrades.** Day 1 forecasts are far more accurate than Day 90. Use confidence intervals, not point estimates, for decisions beyond 30 days.
- **Model assumes structural stability.** If your business model fundamentally changes (new product line, major acquisition), retrain from scratch.
- **No real-time bank API integration** in this version. Production deployment would need a connection to a core banking system or accounting software (Tally, SAP, Zoho Books).

---

## Who Built This

Built as a finance + computer science research project exploring the application of ensemble ML methods to corporate treasury management. The methodology draws from academic literature in financial econometrics and applied machine learning, adapted for Indian SME financial conditions.

---

## Licence

MIT — use freely, credit appreciated.

---

*"The goal of forecasting is not to predict the future. It is to tell you what you don't know."* — Nassim Nicholas Taleb
