# Modern Portfolio Theory Project Documentation

## 1. Project Overview

This project is a practical implementation of Modern Portfolio Theory (MPT) using Python.
It fetches market data from Yahoo Finance, computes return and risk statistics, optimizes
portfolio weights, and visualizes results.

The codebase focuses on:

- understanding return-risk tradeoffs,
- finding optimal portfolios under constraints,
- comparing assets through return and correlation analysis,
- presenting results with charts and allocation tables.


## 2. Repository Structure

- `README.md`
  - Introductory explanation of MPT concepts and a simple toy example.

- `requirements.txt`
  - Python dependencies (numerical computing, optimization, plotting, Jupyter, and
    Plotly static export support through `kaleido`).

- `main.py`
  - Main end-to-end portfolio optimization script.
  - Downloads historical data, computes efficient frontier, marks max-Sharpe and
    minimum-volatility portfolios, calculates Sortino ratio, and builds an interactive
    Plotly figure with a summary table.

- `mpt.py`
  - CLI-style script where a user enters tickers and current weights.
  - Computes provided portfolio metrics and compares them against optimized
    max-Sharpe and minimum-volatility portfolios.

- `returns.py`
  - Standalone annualized return analysis for a selected stock list.
  - Generates a Plotly bar chart of annualized returns.

- `correlation.py`
  - Standalone correlation analysis.
  - Builds daily returns and plots a Seaborn heatmap of correlation matrix.

- `main.ipynb`
  - Notebook with conceptual notes (risk types, covariance/correlation formulas)
    and ticker sanity checks with `yfinance`.

- Generated artifacts in repository root
  - `2 Yrs.png` and `Correlation Matrix.png` appear to be exported visual outputs.


## 3. Core Workflow

### Step 1: Data collection

All analytical scripts fetch adjusted close prices from Yahoo Finance:

- Source: `yfinance.download(..., auto_adjust=True)['Close']`
- Frequency: daily data
- Typical horizons used:
  - 2 years in `main.py`
  - 5 years in `mpt.py`, `returns.py`, and `correlation.py`

### Step 2: Return series construction

Daily returns are computed as percentage changes:

- `daily_returns = prices.pct_change(...).dropna()`

### Step 3: Statistical estimates

From daily returns, the project computes:

- mean daily return vector,
- covariance matrix,
- correlation matrix (for correlation visualization).

### Step 4: Portfolio optimization

The project applies constrained optimization (SciPy SLSQP) to solve:

- maximum Sharpe ratio portfolio,
- minimum variance portfolio,
- efficient frontier points for target returns.

### Step 5: Visualization and reporting

Outputs include:

- efficient frontier chart with random portfolios,
- highlighted optimal portfolios,
- allocation table,
- annualized returns bar chart,
- correlation heatmap.


## 4. Mathematical Concepts and Formulas

### 4.1 Daily return

For asset price series $P_t$:

$$
r_t = \frac{P_t - P_{t-1}}{P_{t-1}}
$$

### 4.2 Annualized expected return

Using mean daily return $\mu_d$ and ~252 trading days:

$$
\mu_{annual} = \mu_d \times 252
$$

For portfolio weights $w$ and asset mean returns vector $\mu$:

$$
E(R_p) = w^T\mu \times 252
$$

### 4.3 Portfolio variance and volatility

With covariance matrix $\Sigma$:

$$
\sigma_p^2 = w^T\Sigma w \times 252
$$

$$
\sigma_p = \sqrt{\sigma_p^2}
$$

### 4.4 Sharpe ratio

With risk-free rate $r_f$:

$$
\text{Sharpe} = \frac{E(R_p) - r_f}{\sigma_p}
$$

In this project, `risk_free_rate` defaults to 0 in optimization functions.

### 4.5 Sortino ratio

Sortino replaces total volatility with downside volatility relative to MAR
(Minimum Acceptable Return):

$$
\text{Sortino} = \frac{E(R_p) - MAR}{\sigma_{downside}}
$$

In `main.py`, MAR is derived from NIFTY50 average daily return (`^NSEI`) and
then annualized inside the Sortino computation.


## 5. Optimization Setup in This Project

### Objective functions

- Max Sharpe: minimize negative Sharpe ratio.
- Min variance: minimize $w^T\Sigma w$.
- Efficient frontier point: minimize variance subject to a target return.

### Constraints

- Full investment constraint: $\sum_i w_i = 1$.

### Bounds

- In `main.py`: default per-asset bound is 5% to 20% (`(0.05, 0.2)`).
- In `mpt.py`: default per-asset bound is 0% to 100% (`(0, 1)`).

This means `main.py` enforces diversification-like minimum/maximum allocations,
while `mpt.py` allows broader unconcentrated-to-concentrated allocations.


## 6. Script-by-Script Explanation

### 6.1 `main.py`

Main responsibilities:

- fetch stock and benchmark data,
- compute optimization outputs,
- generate random portfolios and filter points under frontier,
- visualize all results in a two-column Plotly layout:
  - chart panel (frontier + portfolios + markers),
  - table panel (asset allocations and summary metrics),
- annotate Sharpe and Sortino ratio,
- attempt PNG export using `kaleido`.

Notable implementation details:

- `random_portfolios()` samples weights from a Dirichlet distribution.
- `filter_random_portfolios()` uses interpolation to keep only portfolios below
  the efficient frontier for cleaner visualization.
- Max-Sharpe and Min-Vol allocations are stored as percentages in DataFrames
  for display.

### 6.2 `mpt.py`

Main responsibilities:

- interactive user input for tickers and initial weights,
- validation of weight length and sum-to-100 rule,
- performance metrics for user-provided portfolio,
- optimized max-Sharpe and min-vol alternatives.

This is useful for quickly evaluating a user-defined portfolio against
optimization baselines.

### 6.3 `returns.py`

Main responsibilities:

- compute annualized returns for a hardcoded asset list,
- print returns to terminal,
- render a Plotly bar chart with percentage labels.

### 6.4 `correlation.py`

Main responsibilities:

- compute daily returns from downloaded prices,
- derive return correlation matrix,
- display Seaborn heatmap.

This is useful for diversification analysis because lower correlations generally
improve risk reduction potential.

### 6.5 `main.ipynb`

The notebook acts as a concept notebook rather than a full pipeline.
It contains:

- explanatory markdown on risk and diversification,
- covariance and correlation definitions,
- simple data checks for specific tickers (`SBIGETS.NS`, `KOTAKGOLD.NS`, `USO`).


## 7. Terms and Concepts Used

- Asset: an investable instrument (stock, ETF, etc.).
- Ticker: market symbol used to fetch data (for example `INFY.NS`).
- Return: percentage change in price over a period.
- Expected return: estimated average return (historically proxied here).
- Variance: dispersion of returns around mean.
- Volatility: standard deviation of returns (risk proxy).
- Covariance: co-movement measure between two assets.
- Correlation: normalized covariance in [-1, 1].
- Diversification: mixing assets to reduce unsystematic risk.
- Efficient frontier: set of portfolios with best return for each risk level.
- Sharpe ratio: excess return per unit of total volatility.
- Sortino ratio: excess return per unit of downside volatility.
- MAR: minimum acceptable return threshold in Sortino calculation.
- Risk-free rate: baseline return from a nearly riskless asset.
- SLSQP: constrained nonlinear optimizer used by SciPy.


## 8. How to Run

From the repository root:

1. Create and activate a virtual environment (already present as `.venv` in this
   workspace).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run scripts as needed:

```bash
python correlation.py
python returns.py
python mpt.py
python main.py
```

Notes:

- Internet connection is required for `yfinance` downloads.
- Plotly static image export requires `kaleido` (already listed).
- `.NS` tickers correspond to NSE-listed instruments.


## 9. Assumptions and Limitations

- Historical mean and covariance are treated as forward estimates.
- Annualization uses a fixed 252 trading days.
- No transaction costs, slippage, taxes, or liquidity constraints.
- No regime-switching or time-varying volatility model.
- Results are sensitive to lookback window, ticker set, and missing data.
- Different scripts use different constraints/horizons, so outputs are not
  directly comparable unless aligned.


## 10. Suggested Next Improvements

- Centralize duplicated portfolio functions into a shared utility module.
- Add robust error handling for missing or partially unavailable ticker data.
- Parameterize constraints and date windows via CLI arguments.
- Include backtesting for rebalancing strategies.
- Add unit tests for optimization and metric calculations.
- Add benchmark comparison beyond NIFTY50 and include risk-free proxy data.
