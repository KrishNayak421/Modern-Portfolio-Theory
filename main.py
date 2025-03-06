import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import scipy.optimize as sco  
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt    

# Fetch stock data and calculate returns
def get_data(stocks, start, end):
    stock_data = yf.download(stocks, start=start, end=end, auto_adjust=True)['Close']
    daily_returns = stock_data.pct_change(fill_method=None).dropna()
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()
    return daily_returns, mean_returns, cov_matrix

# Calculate portfolio performance metrics (annualized)
def portfolio_performance(weights, mean_returns, cov_matrix):
    annual_return = np.dot(mean_returns, weights) * 252  # Annualized return (fraction)
    port_variance = np.dot(weights.T, np.dot(cov_matrix, weights)) * 252
    port_volatility = np.sqrt(port_variance)
    return annual_return, port_volatility

# Optimization functions
def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0):
    ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(ret - risk_free_rate) / vol

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate=0, constraint_set=(0.05, 0.2)):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple(constraint_set for _ in range(num_assets))
    result = sco.minimize(
        negative_sharpe_ratio,
        num_assets * [1./num_assets],
        args=args,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result

def portfolio_variance(weights, mean_returns, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights)) * 252

def min_variance(mean_returns, cov_matrix, constraint_set=(0.05, 0.2)):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple(constraint_set for _ in range(num_assets))
    result = sco.minimize(
        portfolio_variance,
        num_assets * [1./num_assets],
        args=args,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints
    )
    return result

def efficient_optimization(mean_returns, cov_matrix, target_return, constraint_set=(0.05, 0.2)):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: np.dot(mean_returns, x) * 252 - target_return})
    bounds = tuple(constraint_set for _ in range(num_assets))
    result = sco.minimize(
        portfolio_variance, num_assets * [1./num_assets], args=args, method='SLSQP',bounds=bounds, constraints=constraints)
    return result

# Generate random portfolios for scatter plotting
def random_portfolios(mean_returns, cov_matrix, num_portfolios=10000, risk_free_rate=0):
    num_assets = len(mean_returns)
    results = np.zeros((3, num_portfolios))
    weights = np.random.dirichlet(np.ones(num_assets), num_portfolios)
    for i in range(num_portfolios):
        ret, vol = portfolio_performance(weights[i], mean_returns, cov_matrix)
        results[0, i] = ret * 100
        results[1, i] = vol * 100
        results[2, i] = (ret - risk_free_rate) / vol
    return results

# Filter random portfolios to lie below the efficient frontier
def filter_random_portfolios(random_rets, random_vols, ef_vols, target_rets):
    ef_interp = interp1d(ef_vols, target_rets, kind='linear', fill_value='extrapolate')
    filtered_rets = []
    filtered_vols = []
    for ret, vol in zip(random_rets, random_vols):
        if ret <= ef_interp(vol):
            filtered_rets.append(ret)
            filtered_vols.append(vol)
    return np.array(filtered_rets), np.array(filtered_vols)

# Calculate optimization results and efficient frontier
def calculate_results(mean_returns, cov_matrix, risk_free_rate=0):
    max_sharpe_obj = max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate)
    sr_ret_frac, sr_vol_frac = portfolio_performance(max_sharpe_obj.x, mean_returns, cov_matrix)
    sr_ret = sr_ret_frac * 100
    sr_vol = sr_vol_frac * 100
    sr_alloc = pd.DataFrame(max_sharpe_obj.x, index=mean_returns.index, columns=['allocation'])
    sr_alloc.allocation = [round(i * 100, 2) for i in sr_alloc.allocation]
    
    min_vol_obj = min_variance(mean_returns, cov_matrix)
    mv_ret_frac, mv_vol_frac = portfolio_performance(min_vol_obj.x, mean_returns, cov_matrix)
    mv_ret = mv_ret_frac * 100
    mv_vol = mv_vol_frac * 100
    mv_alloc = pd.DataFrame(min_vol_obj.x, index=mean_returns.index, columns=['allocation'])
    mv_alloc.allocation = [round(i * 100, 2) for i in mv_alloc.allocation]
    
    target_returns_frac = np.linspace(mv_ret_frac, sr_ret_frac * 1.2, 50)
    efficient_frontier = []
    for ret in target_returns_frac:
        eff = efficient_optimization(mean_returns, cov_matrix, ret)
        eff_vol = np.sqrt(eff.fun) * 100
        efficient_frontier.append(eff_vol)
    target_returns = target_returns_frac * 100
    return sr_ret, sr_vol, sr_alloc, mv_ret, mv_vol, mv_alloc, efficient_frontier, target_returns

# Calculate the Sortino Ratio for a portfolio using a given MAR (daily minimum acceptable return)
def sortino_ratio(weights, daily_returns, MAR):
    portfolio_daily = daily_returns.dot(weights)
    downside_returns = portfolio_daily[portfolio_daily < MAR]
    if len(downside_returns) == 0:
        return np.nan
    downside_std = np.std(downside_returns)
    annual_downside = downside_std * np.sqrt(252)
    annual_portfolio_return = portfolio_daily.mean() * 252
    return (annual_portfolio_return - MAR * 252) / annual_downside

# Create interactive efficient frontier plot with annotations using Plotly
def plot_efficient_frontier(daily_returns, mean_returns, cov_matrix, MAR, risk_free_rate=0):
    sr_ret, sr_vol, sr_alloc, mv_ret, mv_vol, mv_alloc, ef_vols, target_rets = calculate_results(mean_returns, cov_matrix, risk_free_rate)
    
    random_results = random_portfolios(mean_returns, cov_matrix, num_portfolios=20000, risk_free_rate=risk_free_rate)
    random_rets = random_results[0]
    random_vols = random_results[1]
    
    filtered_rets, filtered_vols = filter_random_portfolios(random_rets, random_vols, ef_vols, target_rets)
    
    ef_trace = go.Scatter(
        x=ef_vols,
        y=target_rets,
        mode='lines',
        line=dict(color='darkblue', width=3, dash='dot'),
        name='Efficient Frontier'
    )
    max_sharpe_trace = go.Scatter(
        x=[sr_vol],
        y=[sr_ret],
        mode='markers',
        marker=dict(size=14, color='red', symbol='star', line=dict(width=2, color='black')),
        name='Max Sharpe Ratio'
    )
    min_vol_trace = go.Scatter(
        x=[mv_vol],
        y=[mv_ret],
        mode='markers',
        marker=dict(size=14, color='green', symbol='diamond', line=dict(width=2, color='black')),
        name='Min Volatility'
    )
    random_trace = go.Scatter(
        x=filtered_vols,
        y=filtered_rets,
        mode='markers',
        marker=dict(size=4, color='rgba(0, 100, 255, 0.3)'),
        name='Random Portfolios'
    )
    
    stock_names = mean_returns.index.tolist()
    max_sharpe_allocs = [f"{x:.2f}%" for x in sr_alloc['allocation']]
    min_vol_allocs = [f"{x:.2f}%" for x in mv_alloc['allocation']]
    max_sharpe_column = max_sharpe_allocs + [f"{sr_ret:.2f}%", f"{sr_vol:.2f}%"]
    min_vol_column = min_vol_allocs + [f"{mv_ret:.2f}%", f"{mv_vol:.2f}%"]
    stock_column = stock_names + ['Annual Return', 'Annual Volatility']
    
    table_trace = go.Table(
        header=dict(
            values=['<b>Stock</b>', '<b>Max Sharpe (%)</b>', '<b>Min Vol (%)</b>'],
            fill_color='lightgrey',
            align='left',
            font=dict(size=12, color='black')
        ),
        cells=dict(
            values=[stock_column, max_sharpe_column, min_vol_column],
            align='left',
            fill_color='white',
            font=dict(size=11)
        )
    )
    
    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.6, 0.4],
        specs=[[{"type": "xy"}, {"type": "table"}]],
        horizontal_spacing=0.05
    )
    
    for trace in [random_trace, ef_trace, max_sharpe_trace, min_vol_trace]:
        fig.add_trace(trace, row=1, col=1)
    fig.add_trace(table_trace, row=1, col=2)
    
    sharpe_val = sr_ret / sr_vol
    
    # Convert max Sharpe portfolio weights from percentage to fractions
    max_weights = sr_alloc['allocation'].values / 100  
    sortino_val = sortino_ratio(max_weights, daily_returns, MAR)
    
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.05, y=0.95,
        text=f"Max Sharpe Ratio: {sharpe_val:.2f}",
        showarrow=False,
        font=dict(size=14, color="black"),
        align="left",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        bgcolor="lightgrey",
        opacity=0.8
    )
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.05, y=0.90,
        text=f"Sortino Ratio: {sortino_val:.2f}" if not np.isnan(sortino_val) else "Sortino Ratio: N/A",
        showarrow=False,
        font=dict(size=14, color="black"),
        align="left",
        bordercolor="black",
        borderwidth=1,
        borderpad=4,
        bgcolor="lightgrey",
        opacity=0.8
    )
    
    if len(filtered_vols) == 0 or len(filtered_rets) == 0:
        x_range = [min(random_vols)*0.9, max(random_vols)*1.1]
        y_range = [min(random_rets)*0.9, max(random_rets)*1.1]
    else:
        x_range = [min(filtered_vols)*0.9, max(filtered_vols)*1.1]
        y_range = [min(filtered_rets)*0.9, max(filtered_rets)*1.1]
    
    fig.update_xaxes(range=x_range, row=1, col=1)
    fig.update_yaxes(range=y_range, row=1, col=1)
    
    fig.update_layout(
        title=dict(text='Portfolio Optimization with Modern Portfolio Theory (2-Year Data)',
                   x=0.5, xanchor='center', font=dict(size=24)),
        xaxis=dict(title='Annualized Volatility (%)', gridcolor='lightgray', zeroline=False, tickformat=".2f"),
        yaxis=dict(title='Annualized Returns (%)', gridcolor='lightgray', zeroline=False, tickformat=".2f"),
        legend=dict(x=0.65, y=0.15, bgcolor='rgba(255,255,255,0.7)'),
        width=1600,
        height=800,
        template='plotly_white'
    )
    
    fig.show()

# Example usage
if __name__ == "__main__":
    stock_list = [
        "HDFCBANK.NS",
        "INFY.NS",
        "BAJFINANCE.NS",
        "INDUSINDBK.NS",
        "BALRAMCHIN.NS",
        "INDUSTOWER.NS",
        "CIPLA.NS",
        "KPIGREEN.NS",
        "DHANBANK.NS",
        "SETFNIF50.NS",
        "NIFTYBEES.NS",
        "CPSEETF.NS"
    ]
    
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=730)  # Approximately 2 years of data
    
    # Get portfolio data
    daily_returns, mean_returns, cov_matrix = get_data(stock_list, start_date, end_date)
    
    # Fetch NIFTY50 data using its ticker "^NSEI"
    nifty_data = yf.download("^NSEI", start=start_date, end=end_date, auto_adjust=True)['Close']
    nifty_daily = nifty_data.pct_change().dropna()
    # Force MAR to be a scalar by squeezing the result
    nifty_avg_daily_return = nifty_daily.mean().squeeze()
    
    # Debug prints (optional)
    sr_ret, sr_vol, sr_alloc, mv_ret, mv_vol, mv_alloc, ef_vols, target_rets = calculate_results(mean_returns, cov_matrix)
    print("Max Sharpe Ratio Portfolio Allocations:")
    print(sr_alloc)
    print("\nMin Volatility Portfolio Allocations:")
    print(mv_alloc)
    
    # Plot the efficient frontier using NIFTY50's average daily return as the MAR
    plot_efficient_frontier(daily_returns, mean_returns, cov_matrix, MAR=nifty_avg_daily_return)
