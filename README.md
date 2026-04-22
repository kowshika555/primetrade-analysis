# 📈 PrimeTrade Analysis
### Crypto Trading Performance Analytics Platform

> Analyze cryptocurrency trading performance against Bitcoin Fear & Greed sentiment data — with machine learning, statistical analysis, and an interactive dashboard.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-red?logo=streamlit)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/ML-XGBoost-orange)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## 📋 Table of Contents

1. [Overview](#-overview)
2. [Features](#-features)
3. [Technology Stack](#-technology-stack)
4. [Project Structure](#-project-structure)
5. [Installation](#-installation)
6. [Usage](#-usage)
7. [Key Metrics](#-key-metrics)
8. [Machine Learning Model](#-machine-learning-model)
9. [Visualizations](#-visualizations)
10. [Data Format Requirements](#-data-format-requirements)
11. [Troubleshooting](#-troubleshooting)
12. [Sample Output](#-sample-output)
13. [Contributing](#-contributing)
14. [License](#-license)
15. [Authors](#-authors)
16. [Acknowledgments](#-acknowledgments)
17. [Contact](#-contact)
18. [Disclaimer](#-disclaimer)

---

## 🔭 Overview

**PrimeTrade Analysis** is a data-driven trading analytics platform built for Hyperliquid trader performance research. It merges historical trade data with the Bitcoin Fear & Greed Index to uncover how market sentiment shapes trader behavior and profitability.

The platform answers three core questions:

- 📉 **Do traders perform differently on Fear vs. Greed days?**
- 🔄 **Does market sentiment change trading behavior** — frequency, leverage, long/short bias?
- 🤖 **Can we predict next-day profitability** using sentiment + behavioral features?

It covers the full analytics pipeline: raw data ingestion → cleaning → feature engineering → exploratory analysis → statistical testing → machine learning → interactive visualization.

---

## ✨ Features

### 🔁 Automated Data Processing
- Smart timestamp detection — handles Unix milliseconds, Unix seconds, and date strings automatically
- Flexible column name resolution across different CSV schema variants
- Automatic deduplication, null-filling, and type casting
- Robust merging of trade data with Fear/Greed index on daily granularity

### 📊 Performance Metrics
- Daily PnL aggregation per trader account
- Win rate, loss rate, and win/loss ratio computation
- Sharpe ratio and profit factor calculation
- Max drawdown proxy using cumulative PnL tracking
- Long/short ratio and trade frequency analysis

### 🌡️ Market Sentiment Integration
- Aligns Hyperliquid trade data with Bitcoin Fear & Greed Index
- Groups Extreme Fear → Fear and Extreme Greed → Greed for cleaner binary analysis
- Segment-level breakdown of behavior differences across sentiment regimes

### 🤖 Machine Learning Models
- XGBoost classifier for next-day PnL direction prediction
- 5-fold cross-validation with accuracy reporting
- Feature importance visualization
- Confusion matrix and classification report output

### 📱 Interactive Streamlit Dashboard
- Date range, sentiment, and leverage filters in the sidebar
- KPI cards: total rows, Fear/Greed day counts, average PnL comparison
- Four tabs: Performance, Behavior, Segments, Raw Data
- CSV download of any filtered view

### 📐 Statistical Analysis
- Side-by-side summary statistics (mean, median, std) by sentiment
- KDE plots for leverage distribution
- Violin plots for behavioral metric distributions
- Win rate heatmaps across trader segments

---

## 🛠️ Technology Stack

| Category | Library | Version | Purpose |
|---|---|---|---|
| **Core** | Python | 3.10+ | Runtime |
| **Data** | Pandas | 2.0+ | Data loading, cleaning, aggregation |
| **Data** | NumPy | 1.24+ | Numerical operations |
| **Visualization** | Matplotlib | 3.7+ | Base charting |
| **Visualization** | Seaborn | 0.12+ | Statistical plots |
| **Visualization** | Plotly | 5.15+ | Interactive charts |
| **ML** | Scikit-learn | 1.3+ | Clustering, scaling, PCA, metrics |
| **ML** | XGBoost | 2.0+ | Gradient boosted classifier |
| **Dashboard** | Streamlit | 1.30+ | Interactive web app |
| **Stats** | SciPy | 1.10+ | Statistical tests |
| **Stats** | Statsmodels | 0.14+ | Regression, time series |

---

## 📁 Project Structure

```
primetrade-analysis/
│
├── 📂 data/
│   ├── historical_data.csv          ← Hyperliquid trade history
│   └── fear_greed_index.csv         ← Bitcoin Fear/Greed index
│
├── 📂 outputs/
│   ├── merged_daily.csv             ← Generated: daily metrics + sentiment
│   ├── account_profile.csv          ← Generated: per-account segment profiles
│   ├── 01_performance_by_sentiment.png
│   ├── 02_pnl_distribution.png
│   ├── 03_behavior_by_sentiment.png
│   ├── 04_pnl_timeline.png
│   ├── 05_segment_pnl_by_sentiment.png
│   ├── 06_winrate_heatmap.png
│   ├── 07_leverage_distribution.png
│   ├── 08_ml_model.png
│   ├── 09_elbow_method.png
│   └── 10_clustering_pca.png
│
├── 📓 analysis.ipynb                ← Main notebook (Parts A, B, C + Bonus)
├── 🖥️  streamlit_app.py             ← Interactive dashboard
├── 📄 requirements.txt              ← All Python dependencies
└── 📖 README.md
```

> 📌 **Note:** The `outputs/` folder and all `.png` charts are generated automatically when you run all cells in `analysis.ipynb`. Do not create them manually.

---

## ⚙️ Installation

### Prerequisites

- **Python 3.10 or higher** — [Download](https://www.python.org/downloads/)
- **Git** — [Download](https://git-scm.com/downloads)
- **VS Code** (recommended) — [Download](https://code.visualstudio.com/)

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/your-username/primetrade-analysis.git
cd primetrade-analysis
```

---

### Step 2 — Create a virtual environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Mac / Linux
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt.

---

### Step 3 — Install dependencies

```bash
# Standard install
pip install -r requirements.txt
```

> ⚠️ **Windows Device Guard / Managed PC Notice:**
> If you see `"This program is blocked by group policy"` or pip fails silently, use this instead:
> ```bash
> python -m pip install -r requirements.txt
> ```
> Using `python -m pip` bypasses Device Guard restrictions that block the standalone `pip.exe`. If you are on a corporate or college machine and pip still fails, try:
> ```bash
> python -m pip install -r requirements.txt --user
> ```

---

### Step 4 — Register the Jupyter kernel

```bash
python -m ipykernel install --user --name=primetrade --display-name "PrimeTrade (venv)"
```

---

### Step 5 — Add your data files

Place your CSV files inside the `data/` folder:

```
data/historical_data.csv       ← Hyperliquid trade export
data/fear_greed_index.csv      ← Fear/Greed index (see format below)
```

---

### Step 6 — Verify installation

```bash
python -c "import pandas, numpy, seaborn, matplotlib, sklearn, xgboost, streamlit; print('All packages OK ✅')"
```

---

## 🚀 Usage

### Option 1 — Jupyter Notebook (Full Analysis)

1. Open VS Code and navigate to the project folder
2. Open `analysis.ipynb`
3. Click the kernel selector (top right) → choose **PrimeTrade (venv)**
4. Run all cells: `Ctrl + Shift + P` → **"Notebook: Run All Cells"**

The notebook runs in this order:

```
Cell 1  → Imports & config
Cell 2  → Load raw CSVs
Cell 3  → Data quality report
Cell 4  → Parse timestamps & clean fear/greed
Cell 5  → Normalise columns & engineer daily metrics
Cell 6  → Merge datasets (creates df)
Cells 7–16  → Part B analysis + 7 charts
Cell 17 → Part C strategy rules
Cells 18–20 → XGBoost ML model + charts
Cells 21–22 → KMeans clustering + PCA
Cell 23 → Export outputs/ CSVs  ← must complete before dashboard
```

---

### Option 2 — Streamlit Dashboard

> ⚠️ You **must** run the notebook through Cell 23 first. The dashboard reads `outputs/merged_daily.csv` and `outputs/account_profile.csv`.

```bash
# Make sure venv is active
venv\Scripts\activate

# Launch dashboard
streamlit run streamlit_app.py
```

Then open your browser at: **http://localhost:8501**

The dashboard includes:
- **📈 Performance tab** — PnL bars, distribution box plot, timeline
- **🧠 Behavior tab** — Violin plots, leverage KDE, trade frequency
- **🔵 Segments tab** — Segment PnL charts, win rate heatmap, cluster scatter
- **📋 Raw data tab** — Filtered table + CSV download

---

### Option 3 — Generate charts only (no notebook UI)

```bash
# Run the notebook headlessly from terminal
jupyter nbconvert --to notebook --execute analysis.ipynb --output analysis_executed.ipynb
```

Charts will be saved to `outputs/` automatically.

---

## 📏 Key Metrics

The following metrics are computed daily per account and aggregated for analysis:

| Metric | Description | Formula |
|---|---|---|
| **Total P&L** | Sum of closed profit/loss for the day | `Σ closedPnL` |
| **Win Rate** | Fraction of trades that closed profitable | `winning_trades / total_trades` |
| **Sharpe Ratio** | Risk-adjusted return (annualised) | `mean(PnL) / std(PnL) × √252` |
| **Max Drawdown** | Largest peak-to-trough drop in cumulative PnL | `cum_pnl - peak_cum_pnl` |
| **Win/Loss Ratio** | Average win size vs. average loss size | `avg_win / avg_loss` |
| **Profit Factor** | Gross profit divided by gross loss | `Σ gains / Σ losses` |
| **Long/Short Ratio** | Proportion of long trades in total | `long_trades / total_trades` |
| **Avg Leverage** | Mean leverage used across all trades | `mean(leverage)` |

---

## 🤖 Machine Learning Model

### Model: XGBoost Classifier

The model predicts whether a trader's **next-day PnL will be positive (1) or negative (0)**.

### Features Used

| Feature | Type | Description |
|---|---|---|
| `total_pnl` | Continuous | Today's total PnL |
| `win_rate` | Continuous | Today's win rate |
| `avg_leverage` | Continuous | Average leverage today |
| `trade_count` | Integer | Number of trades today |
| `long_ratio` | Continuous | Fraction of long trades |
| `avg_size` | Continuous | Average position size |
| `pnl_per_trade` | Continuous | PnL divided by trade count |
| `sentiment_enc` | Binary | 1 = Greed day, 0 = Fear day |

### Hyperparameters

```python
XGBClassifier(
    n_estimators     = 200,
    max_depth        = 4,
    learning_rate    = 0.05,
    subsample        = 0.8,
    colsample_bytree = 0.8,
    eval_metric      = 'logloss',
    random_state     = 42,
)
```

### Expected Performance

| Metric | Expected Range |
|---|---|
| 5-fold CV Accuracy | **65% – 75%** |
| Precision (Profit class) | 0.62 – 0.72 |
| Recall (Profit class) | 0.60 – 0.70 |
| F1 Score | 0.61 – 0.71 |

> 📌 Performance varies depending on dataset size, date range, and asset distribution. The sentiment feature (`sentiment_enc`) typically ranks in the top 3 most important features.

---

## 📊 Visualizations

The notebook generates 10 charts, all saved to `outputs/`:

| File | Chart | Description |
|---|---|---|
| `01_performance_by_sentiment.png` | Bar charts | Avg PnL, win rate, and leverage — Fear vs Greed |
| `02_pnl_distribution.png` | Box + strip plot | Daily PnL spread across sentiment classes |
| `03_behavior_by_sentiment.png` | Violin plots | Trade frequency, leverage, and long ratio distributions |
| `04_pnl_timeline.png` | Scatter plot | Daily PnL over time, coloured by sentiment |
| `05_segment_pnl_by_sentiment.png` | Grouped bar | Avg PnL for 3 trader segments × 2 sentiments |
| `06_winrate_heatmap.png` | Heatmap | Win rate by segment and sentiment |
| `07_leverage_distribution.png` | KDE plot | Leverage density curves — Fear vs Greed |
| `08_ml_model.png` | Feature importance + confusion matrix | XGBoost model evaluation |
| `09_elbow_method.png` | Line chart | KMeans inertia vs K for cluster selection |
| `10_clustering_pca.png` | PCA scatter | Trader archetypes projected to 2D |

---

## 📂 Data Format Requirements

### `data/historical_data.csv` (Hyperliquid trades)

The column resolver accepts multiple naming variants automatically:

| Standard Name | Accepted Column Names |
|---|---|
| `account` | `account`, `Account`, `trader`, `address`, `user`, `wallet` |
| `pnl` | `closedPnL`, `closed_pnl`, `closedPL`, `pnl`, `PnL`, `profit`, `realizedPnl` |
| `size` | `sz`, `size`, `Size`, `qty`, `quantity`, `amount`, `notional` |
| `side` | `side`, `Side`, `direction`, `type`, `tradeType`, `dir` |
| `leverage` | `leverage`, `Leverage`, `lev`, `maxLeverage` |
| `time` | `time`, `timestamp`, `Time`, `Timestamp`, `date`, `created_at` |
| `symbol` | `symbol`, `Symbol`, `coin`, `asset`, `market`, `ticker` |

**Timestamp formats supported:**
- Unix milliseconds (e.g. `1704067200000`)
- Unix seconds (e.g. `1704067200`)
- Date strings (e.g. `2024-01-01`, `01/01/2024`)

**Side values supported:**
`BUY`, `SELL`, `LONG`, `SHORT`, `A`, `B`, `BID`, `ASK`, `1`, `-1`

---

### `data/fear_greed_index.csv` (Fear & Greed Index)

| Standard Name | Accepted Column Names |
|---|---|
| `date` | `date`, `Date`, `timestamp`, `Timestamp`, `time` |
| `sentiment` | `Classification`, `classification`, `sentiment`, `Sentiment`, `label`, `value_classification`, `category` |

**Sentiment values supported:**
`Fear`, `Greed`, `Extreme Fear` (→ grouped to Fear), `Extreme Greed` (→ grouped to Greed)

**Minimum required columns:** `date` + `sentiment`

---

## 🔧 Troubleshooting

### ❌ Device Guard blocks pip on Windows

**Error:** `"This program is blocked by group policy. For more information, contact your system administrator."`

**Fix:** Always invoke pip through Python itself:
```bash
python -m pip install -r requirements.txt
```
If you are on a managed corporate/college machine, add `--user`:
```bash
python -m pip install -r requirements.txt --user
```

---

### ❌ Streamlit import errors (`ModuleNotFoundError`)

**Error:** `ModuleNotFoundError: No module named 'seaborn'`

**Fix:** The module is missing from your virtual environment. Run:
```bash
venv\Scripts\activate
python -m pip install seaborn matplotlib pandas numpy scikit-learn xgboost streamlit
```

Then verify:
```bash
python -c "import seaborn; print(seaborn.__version__)"
```

> ⚠️ Never use the auto-install trick (`subprocess.check_call`) inside Streamlit apps on restricted networks. It causes a crash loop when pip times out. Install manually instead.

---

### ❌ KDE plot error (`ValueError: array must not contain NaN`)

**Error:** KDE fails when a sentiment group has only one unique leverage value (zero variance).

**Fix:** Already handled in `streamlit_app.py` with:
```python
if lev.nunique() > 1:
    lev.plot.kde(...)
```
If you are writing custom analysis, apply the same guard before any `.plot.kde()` call.

---

### ❌ Date parsing errors (`KeyError` or wrong dates)

**Fix:** Add this debug cell before Cell 4 in the notebook:
```python
print("Time column sample:", trades_raw.iloc[:, 0].head(5).tolist())
print("All columns:", trades_raw.columns.tolist())
```
Then match the format to one of the three auto-detect branches in Cell 4. If your format is unusual (e.g. `DD-MM-YYYY`), override manually:
```python
trades['date'] = pd.to_datetime(trades[time_col], format='%d-%m-%Y').dt.normalize()
```

---

### ❌ `NameError: name 'df' is not defined`

**Cause:** Cells were not run in order. `df` is created in Cell 6, which depends on `fg` (Cell 4) and `daily` (Cell 5).

**Fix:** Click **"Restart Kernel and Run All Cells"** in VS Code (`Ctrl + Shift + P` → search for it). Always run top-to-bottom.

---

### ❌ `KeyError: 'pnl'` in Cell 5

**Cause:** None of the candidate column names matched the actual PnL column in your CSV.

**Fix:** Add a debug print at the top of Cell 5:
```python
print(trades.columns.tolist())
```
Then add your exact column name to the `'pnl'` candidates list in `COLUMN_CANDIDATES`:
```python
'pnl': ['closedPnL', 'your_actual_column_name', ...],
```

---

## 📋 Sample Output

After running all cells in `analysis.ipynb`, you should see something like:

```
====== KEY INSIGHT SUMMARY ======

── FEAR DAYS (142 unique dates) ──
  Avg total PnL    : -0.0312
  Avg win rate     : 0.4821
  Avg leverage     : 8.74x
  Avg trades/day   : 6.3
  Avg long ratio   : 0.4413
  Avg drawdown     : -128.44

── GREED DAYS (201 unique dates) ──
  Avg total PnL    : 0.1847
  Avg win rate     : 0.5234
  Avg leverage     : 9.12x
  Avg trades/day   : 7.8
  Avg long ratio   : 0.5891
  Avg drawdown     : -84.22
```

```
=== XGBoost Classification Report ===
              precision    recall  f1-score   support
        Loss       0.64      0.61      0.62      1842
      Profit       0.69      0.72      0.70      2203

    accuracy                           0.67      4045
   macro avg       0.67      0.67      0.66      4045

5-fold CV accuracy: 0.681 ± 0.014
```

```
✅ Exported:
   outputs/merged_daily.csv
   outputs/account_profile.csv

📊 Charts saved (10 files):
   outputs/01_performance_by_sentiment.png
   ...
   outputs/10_clustering_pca.png

🎉 ALL CELLS COMPLETE!
   Run the dashboard: streamlit run streamlit_app.py
```

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. **Fork** the repository
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Commit your changes:
   ```bash
   git commit -m "Add: your feature description"
   ```
4. Push to your fork:
   ```bash
   git push origin feature/your-feature-name
   ```
5. Open a **Pull Request** on GitHub

### Contribution Ideas
- Add support for additional exchanges (Binance, Bybit)
- Extend the ML model with LSTM or time-series features
- Add statistical significance tests (t-test, Mann-Whitney U)
- Support for multi-asset portfolio analysis
- Improve dashboard mobile responsiveness

Please follow [PEP 8](https://pep8.org/) style conventions and include docstrings for any new functions.

---

## 📄 License

This project is licensed under the **MIT License**.

```
MIT License

Copyright (c) 2024 PrimeTrade Analysis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
```

See [LICENSE](LICENSE) for the full text.

---

## 👥 Authors

| Name | Role | Contact |
|---|---|---|
| **Kowshika Selvakumar** | Lead Analyst & Developer | [GitHub](https://github.com/your-username) |

Built as part of the **Primetrade.ai Data Science Intern — Round 0 Assignment**.

---

## 🙏 Acknowledgments

- [**Hyperliquid**](https://hyperliquid.xyz/) — for the decentralized trading infrastructure and open trade data
- [**Alternative.me**](https://alternative.me/crypto/fear-and-greed-index/) — for the Bitcoin Fear & Greed Index data
- [**XGBoost Team**](https://xgboost.readthedocs.io/) — for the gradient boosting library
- [**Streamlit**](https://streamlit.io/) — for making interactive ML dashboards accessible
- [**Primetrade.ai**](https://primetrade.ai/) — for designing a genuinely interesting real-world assignment

---

## 📬 Contact

For questions about this project:

- 📧 **Email:** your.email@example.com
- 💼 **LinkedIn:** [linkedin.com/in/your-profile](https://linkedin.com/in/your-profile)
- 🐙 **GitHub:** [github.com/your-username](https://github.com/your-username)

For questions about the internship assignment:
- 📧 **Primetrade.ai:** sonika@primetrade.ai

---

## ⚠️ Disclaimer

> This project is built **for educational and research purposes only**.

- The analysis, insights, and strategy rules generated by this platform are **not financial advice**.
- Past trading performance shown in the data does **not guarantee future results**.
- Cryptocurrency trading involves substantial risk of loss. Do not make trading decisions based solely on this analysis.
- The authors are not responsible for any financial losses incurred as a result of using this software.
- All data used in this project is historical and sourced from publicly available APIs.

---

<div align="center">

**PrimeTrade Analysis** · Built with 🐍 Python & ❤️ for data

*Primetrade.ai Data Science Intern Assignment — Round 0*

</div>"# primetrade-analysis" 
