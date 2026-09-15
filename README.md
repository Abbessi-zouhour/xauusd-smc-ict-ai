
# XAUUSD SMC/ICT AI Research & Backtesting System

An educational quantitative research and backtesting framework for analyzing **XAUUSD (Gold)** using multi-timeframe market structure, liquidity, ICT/SMC concepts, Supply & Demand, and machine learning.

The project aims to investigate whether structured market-context features can be used to identify and evaluate historical trading setups while applying a minimum **2:1 reward-to-risk (R:R)** research filter.

> **Disclaimer:** This project is intended for educational, research, and historical backtesting purposes. It does not provide financial advice, execute live trades, or guarantee future performance.

---

## 🎯 Project Objective

The main objective is to build a systematic research pipeline that combines:

- Multi-timeframe market analysis
- Market structure
- Swing high / swing low detection
- Break of Structure (BOS)
- Market Structure Shift (MSS)
- Liquidity detection
- Liquidity sweep detection
- ICT concepts
- Smart Money Concepts (SMC)
- Supply & Demand
- Feature engineering
- XGBoost machine learning
- Reward-to-risk filtering
- Historical backtesting
- Out-of-sample testing
- Walk-forward validation

The system is designed to separate **market-structure logic** from **machine-learning evaluation**.

The market-structure layer identifies and defines potential setups, while the machine-learning layer evaluates the historical characteristics of those setups.

---

# 🧠 System Architecture

```text
                           XAUUSD
                              │
                              ▼
                       MetaTrader 5
                              │
                              ▼
                         OHLCV Data
                              │
                              ▼
                      Data Quality Check
                              │
                              ▼
                       Data Processing
                              │
                              ▼
                  Multi-Timeframe Context
                              │
             ┌────────────────┼────────────────┐
             │                │                │
             ▼                ▼                ▼
            D1               H4               H1
             │                │                │
             └────────────────┼────────────────┘
                              │
                              ▼
                             M15
                              │
                              ▼
                    Market Structure
                              │
                              ▼
                         Liquidity
                              │
                              ▼
                       ICT / SMC
                              │
                              ▼
                     Supply & Demand
                              │
                              ▼
                       Setup Detection
                              │
                              ▼
                        2R Filter
                              │
                     ┌────────┴────────┐
                     │                 │
                   < 2R               ≥ 2R
                     │                 │
                   Reject               ▼
                                   Feature Engineering
                                         │
                                         ▼
                                      XGBoost
                                         │
                                         ▼
                                  Historical Testing
                                         │
                                         ▼
                                  Out-of-Sample Test
                                         │
                                         ▼
                                  Walk-Forward Test
📊 Current Project Status
Implemented
 Python project structure
 Python virtual environment
 MetaTrader 5 connection
 Exness XAUUSD symbol discovery
 Historical market-data download
 M15 dataset
 H1 dataset
 H4 dataset
 D1 dataset
 Data-quality validation
 Duplicate detection
 Missing-value detection
 OHLC validation
 Timestamp validation
 Swing high detection
 Swing low detection
 Confirmed swing detection
 Basic market-structure detection
 Break of Structure (BOS)
 Market Structure Shift (MSS)
 Previous liquidity levels
 Equal-level detection
 Liquidity sweep detection
 Structure unit tests
 Liquidity unit tests
 Setup unit tests
In Development
 More robust ICT concepts
 Fair Value Gap detection
 Order Block detection
 Premium / Discount zones
 Internal / External liquidity
 Supply & Demand refinement
 Multi-timeframe setup confirmation
 2R setup filtering
 Feature engineering
 XGBoost training pipeline
 Label generation
 Backtesting engine
 Out-of-sample evaluation
 Walk-forward validation
 Performance reporting
 Visualization dashboard
 Experiment tracking
🗂️ Project Structure
xauusd-smc-ict-ai/
│
├── data/
│   ├── raw/
│   │   ├── xauusd_m15.csv
│   │   ├── xauusd_h1.csv
│   │   ├── xauusd_h4.csv
│   │   └── xauusd_d1.csv
│   │
│   └── processed/
│
├── models/
│
├── reports/
│   └── data_quality.csv
│
├── src/
│   ├── __init__.py
│   ├── backtest.py
│   ├── config.py
│   ├── data_loader.py
│   ├── data_quality.py
│   ├── download_data.py
│   ├── features.py
│   ├── ict.py
│   ├── labels.py
│   ├── liquidity.py
│   ├── main.py
│   ├── model.py
│   ├── mt5_data.py
│   ├── setups.py
│   ├── structure.py
│   └── supply_demand.py
│
├── tests/
│   ├── __init__.py
│   ├── test_liquidity.py
│   ├── test_setups.py
│   └── test_structure.py
│
├── .gitignore
├── requirements.txt
└── README.md
📈 Market Data

Historical market data is obtained through MetaTrader 5.

The current broker environment uses:

Symbol: XAUUSDm
Broker: Exness
Platform: MetaTrader 5

The data is stored locally in:

data/raw/

The current datasets are:

data/raw/
├── xauusd_m15.csv
├── xauusd_h1.csv
├── xauusd_h4.csv
└── xauusd_d1.csv
Historical Coverage
Timeframe	Approx. Candles	Approx. Start	Approx. End
M15	63,842	2024	2026
H1	27,779	2022	2026
H4	13,893	2018	2026
D1	2,700	2018	2026

The exact available history depends on the broker's MetaTrader 5 historical data.

🔍 Data Quality

Before market data is used by the research pipeline, it is validated.

The data-quality module checks:

Duplicate timestamps
Missing values
Invalid OHLC relationships
Non-monotonic timestamps
Invalid numerical values

The implementation is located in:

src/data_quality.py

The generated report is stored locally in:

reports/data_quality.csv

The current downloaded datasets pass the basic data-quality checks.

🕐 Multi-Timeframe Analysis

The system uses multiple timeframes to separate higher-level context from lower-timeframe setup formation.

D1 — Daily Context

Used for broad market context.

Potential information includes:

Major swing structure
Major highs and lows
Higher-timeframe liquidity
Broad directional context
H4 — Higher-Timeframe Structure

Used to refine the larger market structure.

Potential information includes:

BOS
MSS
Major swing points
Supply / Demand context
Higher-timeframe liquidity
H1 — Intermediate Structure

Used to analyze intermediate market behavior.

Potential information includes:

Internal structure
Liquidity
Structural confirmation
Setup context
M15 — Setup Formation

Used for lower-timeframe analysis.

Potential information includes:

Liquidity sweeps
MSS
BOS
Setup confirmation
Entry-area structure
🏗️ Market Structure

Market-structure logic is implemented in:

src/structure.py

The current system includes:

Swing highs
Swing lows
Confirmed swings
Break of Structure (BOS)
Market Structure Shift (MSS)
Basic structural direction

A simplified bullish structure example:

        Higher High
             ▲
            / \
           /   \
      Higher Low
          ▲
          │
      Break High
          │
          ▼
         BOS

A simplified bearish structure:

      Lower High
          ▼
          │
      Break Low
          │
          ▼
         BOS
🛡️ Lookahead-Bias Protection

Avoiding lookahead bias is a core requirement of this project.

A historical system must not use information that would only become available in the future.

Swing detection therefore uses confirmation bars.

Conceptually:

Previous candles
       │
       ▼
Left bars → Candidate Swing ← Right bars
                            │
                            ▼
                     Swing confirmed

The swing is only considered confirmed after the required right-side candles have occurred.

Liquidity calculations also use shifted historical values where necessary to prevent the current candle from using information that was not available at that point in time.

💧 Liquidity Detection

Liquidity logic is implemented in:

src/liquidity.py

Current functionality includes:

Previous Highs and Lows

Historical rolling highs and lows are used to identify potential liquidity levels.

Equal Levels

Nearby highs and lows can be grouped using a configurable tolerance.

Liquidity Sweeps

A basic liquidity sweep is detected when price moves beyond a previous level and then closes back through that level.

Conceptual example:

Previous High
────────────────────
             ▲
             │
          Sweep
             │
Price ───────┘
             │
             ▼
       Close below

The current liquidity implementation is a research-oriented baseline and will be refined as the project develops.

🧩 ICT / SMC Layer

ICT/SMC-related logic is implemented through:

src/ict.py
src/setups.py

The long-term research framework is intended to analyze concepts such as:

Liquidity
Buy-side liquidity
Sell-side liquidity
Liquidity sweeps
BOS
MSS
Fair Value Gaps
Order Blocks
Displacement
Premium / Discount
Supply & Demand
Multi-timeframe confluence

The objective is to transform these concepts into measurable and testable rules rather than subjective chart interpretations.

📦 Supply & Demand

Supply and Demand logic is implemented in:

src/supply_demand.py

The purpose is to identify price areas that may contain useful historical information.

Future development will investigate:

Zone identification
Zone strength
Fresh zones
Tested zones
Departure strength
Retests
Multi-timeframe zones
Confluence with liquidity and structure
⚖️ Reward-to-Risk Filter

The research framework uses a minimum:

R:R = 2.0

For example:

Risk   = 10 points
Reward = 20 points

R:R = Reward / Risk
   = 20 / 10
   = 2.0

A setup whose available historical reward is less than 2R can be rejected before entering the machine-learning stage.

The 2R threshold is a research filter, not a guarantee of success.

🤖 Machine Learning

The planned machine-learning component uses XGBoost.

The model implementation is located in:

src/model.py

The objective is not simply to predict whether the next candle will be bullish or bearish.

Instead, the model should evaluate structured setup information.

Potential features include:

Market structure
Liquidity sweep
Swing distance
Higher-timeframe direction
Supply / Demand context
ICT conditions
Volatility
Candle structure
Timeframe alignment
Reward-to-risk

Conceptually:

Market Data
     │
     ▼
Feature Engineering
     │
     ▼
Setup Candidate
     │
     ▼
2R Filter
     │
     ▼
XGBoost
     │
     ▼
Historical Outcome
🏷️ Label Generation

Historical outcome labels are handled in:

src/labels.py

The labeling process will determine whether a historical setup reaches its predefined target or invalidation condition within a defined evaluation window.

Labels must be generated chronologically and without using information from the future beyond the predefined evaluation period.

🧮 Feature Engineering

Feature engineering is implemented in:

src/features.py

The purpose is to transform raw OHLCV and structural information into numerical features suitable for machine learning.

Potential features include:

swing_high
swing_low
bos
mss
liquidity_sweep
distance_to_high
distance_to_low
volatility
trend_context
supply_zone
demand_zone
rr_ratio

The feature set will evolve throughout the research process.

🔬 Backtesting

The backtesting framework is located in:

src/backtest.py

The goal is to evaluate the methodology on historical data.

Potential evaluation metrics include:

Number of setups
Number of accepted setups
Number of rejected setups
Win rate
Loss rate
Average R:R
Maximum drawdown
Profit factor
Expectancy
Sharpe ratio where appropriate
Out-of-sample performance
Performance stability
Performance by market regime

Model accuracy alone is not sufficient to evaluate a trading research system.

🧪 Out-of-Sample Testing

The system should separate training data from unseen evaluation data.

Conceptually:

Historical Data
      │
      ├─────────────────┐
      ▼                 ▼
 Training Data       Test Data
      │                 │
      ▼                 │
 Train Model             │
      │                 │
      └────────┬─────────┘
               ▼
       Unseen Evaluation

The model should never be evaluated using the same observations that were used to train it.

🔄 Walk-Forward Validation

Because financial market data is chronological, walk-forward validation is planned.

Conceptually:

Train ───────► Test

      Train ───────► Test

            Train ───────► Test

                  Train ───────► Test

This allows the model to be repeatedly trained on historical information and evaluated on subsequent unseen periods.

🧪 Testing

The project uses pytest.

Run the test suite with:

pytest -q

Current tests include:

tests/test_structure.py
tests/test_liquidity.py
tests/test_setups.py

The current test suite contains tests for:

Market structure
Swing detection
Liquidity levels
Liquidity sweeps
Setup logic

Example result:

6 passed

The test suite will expand as additional modules are implemented.

⚙️ Installation
Requirements

Recommended environment:

Python 3.11
MetaTrader 5
Windows
Clone the Repository
git clone https://github.com/Abbessi-zouhour/xauusd-smc-ict-ai.git
cd xauusd-smc-ict-ai
Create Virtual Environment
python -m venv .venv
Activate Environment
.\.venv\Scripts\Activate.ps1
Install Dependencies
pip install -r requirements.txt
📥 Download Market Data

The MetaTrader 5 data downloader is located in:

src/mt5_data.py

The system discovers the available XAUUSD symbol and downloads historical OHLCV data.

The current Exness Gold symbol is:

XAUUSDm

The resulting datasets are stored in:

data/raw/

Raw datasets are intentionally excluded from GitHub because they can be large and broker-specific.

🔎 Run Data Quality Validation

From the project root:

python src/data_quality.py

The validation checks the downloaded datasets and generates:

reports/data_quality.csv
🧪 Run Tests
pytest -q
⚙️ Configuration

Project configuration is handled through:

src/config.py

Configuration may include:

Timeframes
Lookback periods
Swing parameters
Liquidity tolerance
Reward-to-risk threshold
Model parameters
Backtesting settings

Keeping configuration separate from implementation makes experimentation easier.

🔐 Git & Data Policy

The repository intentionally does not commit large or sensitive local files.

The following are ignored:

.venv/
data/raw/*.csv
data/processed/*.csv
reports/*.csv
models/*.pkl
models/*.joblib
models/*.json
.env
.env.*

GitHub therefore contains the source code and project configuration while local market data remains on the development machine.

🗺️ Development Roadmap
Phase 1 — Data
 MT5 connection
 XAUUSD symbol discovery
 Historical data download
 M15 / H1 / H4 / D1 datasets
 Data-quality validation
Phase 2 — Market Structure
 Swing detection
 Confirmed swings
 BOS
 MSS
 Internal / external structure refinement
Phase 3 — Liquidity
 Previous highs / lows
 Equal levels
 Basic liquidity sweeps
 Buy-side / sell-side liquidity classification
 Liquidity pools
 Internal / external liquidity
Phase 4 — ICT / SMC
 Fair Value Gaps
 Order Blocks
 Displacement
 Premium / Discount
 Advanced market structure
 Multi-timeframe confluence
Phase 5 — Supply & Demand
 Zone detection
 Zone strength
 Freshness
 Retest logic
 Multi-timeframe zones
Phase 6 — Setup Engine
 Setup definitions
 Entry conditions
 Structural stop placement
 Target calculation
 2R filtering
 Setup scoring
Phase 7 — Machine Learning
 Feature engineering
 Label generation
 XGBoost training
 Hyperparameter tuning
 Feature importance
 Probability calibration
Phase 8 — Validation
 Historical backtesting
 Out-of-sample testing
 Walk-forward validation
 Robustness testing
 Sensitivity analysis
 Performance reporting
Phase 9 — Research Interface
 Setup visualization
 Backtesting dashboard
 Model statistics
 Feature-importance visualization
 Experiment comparison
🧠 Research Principles
No Lookahead Bias

Future market information must never be used to construct features for an earlier timestamp.

Chronological Validation

Training and testing must respect the time-series nature of financial data.

Modular Architecture

Market structure, liquidity, ICT, feature engineering, machine learning, and backtesting remain separated into dedicated modules.

Testable Components

Core detection functions should have automated unit tests.

Research Before Automation

The primary objective is to determine whether the methodology has measurable historical characteristics before considering further automation.

Avoid Overfitting

A strategy should not be judged only by performance on the historical data used to develop it.

No Guaranteed Results

Historical results do not guarantee future performance.

🛠️ Technologies

The project uses:

Python
Pandas
NumPy
Scikit-learn
XGBoost
SciPy
Matplotlib
Seaborn
TA
MetaTrader 5
Pytest
📌 Important Limitations

This research system has several important limitations.

Broker-Specific Data

XAUUSD is an OTC instrument and historical pricing can differ between brokers.

Spread and Slippage

Historical OHLC data does not necessarily represent the exact execution conditions that would occur in a real environment.

Market Regime Changes

Patterns observed in one period may not remain stable in another period.

Overfitting

A model can appear highly effective on historical data while performing poorly on unseen data.

Data Snooping

Testing many combinations of parameters can produce misleading results if the final model is selected based only on historical performance.

Technical Definitions

Concepts such as liquidity, order blocks, market structure, and supply/demand can have subjective definitions. This project therefore aims to convert them into explicit, testable rules.

⚠️ Disclaimer

This repository is an educational quantitative research project.

It is intended for:

Learning
Algorithmic research
Historical market analysis
Backtesting
Machine-learning experimentation

It is not financial advice.

A backtest or machine-learning result does not guarantee future performance.

Research results may be affected by:

Data quality
Broker-specific pricing
Spread
Slippage
Transaction costs
Market regime changes
Overfitting
Lookahead bias
Selection bias
Data snooping
Model instability

All results should therefore be interpreted as experimental research rather than predictions of future market behavior.

👩‍💻 Author
Zouhour Abbassi

AI / Data / Machine Learning Developer

GitHub

https://github.com/Abbessi-zouhour

LinkedIn

https://www.linkedin.com/in/zouhour-abbassi/

Portfolio

https://portfolio-website-up4t.vercel.app/

📄 License

This project is currently intended as a personal research and educational project.