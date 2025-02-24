import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import scipy.optimize as sco
import plotly.graph_objects as go
  
# Fetch stock data and calculate returns
def get_data(stocks, start, end):
    stock_data = yf.download(stocks, start=start, end=end, auto_adjust=True)['Close']
    returns = stock_data.pct_change(fill_method=None)
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return mean_returns, cov_matrix

# Calculate portfolio performance metrics (annualized)
def portfolio_performance(weights, mean_returns, cov_matrix):
    annual_return = np.dot(mean_returns, weights) * 252  # in fraction
    port_variance = np.dot(weights.T, np.dot(cov_matrix, weights)) * 252
    port_volatility = np.sqrt(port_variance)             # in fraction
    return annual_return, port_volatility

# Optimization functions
def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0):
    ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(ret - risk_free_rate) / vol

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate=0, constraint_set=(0,1)):
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

def min_variance(mean_returns, cov_matrix, constraint_set=(0,1)):
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

def efficient_optimization(mean_returns, cov_matrix, target_return, constraint_set=(0,1)):
    # Note: target_return is expected in fraction (e.g., 0.15 for 15%)
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix)
    
    constraints = (
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
        {'type': 'eq', 'fun': lambda x: np.dot(mean_returns, x) * 252 - target_return}
    )
    
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

# Generate random portfolios for scatter plotting
def random_portfolios(mean_returns, cov_matrix, num_portfolios=5000, risk_free_rate=0):
    num_assets = len(mean_returns)
    results = np.zeros((3, num_portfolios))
    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
        # Convert to percentages
        results[0, i] = ret * 100
        results[1, i] = vol * 100
        results[2, i] = (ret - risk_free_rate) / vol
    return results

# Calculate optimization results and efficient frontier
def calculate_results(mean_returns, cov_matrix, risk_free_rate=0):
    # Max Sharpe Ratio Portfolio
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix)
    sr_ret_frac, sr_vol_frac = portfolio_performance(max_sharpe.x, mean_returns, cov_matrix)
    sr_ret = sr_ret_frac * 100  # convert to %
    sr_vol = sr_vol_frac * 100  # convert to %
    sr_alloc = pd.DataFrame(max_sharpe.x, index=mean_returns.index, columns=['allocation'])
    sr_alloc.allocation = [round(i*100, 2) for i in sr_alloc.allocation]
    
    # Min Volatility Portfolio
    min_vol = min_variance(mean_returns, cov_matrix)
    mv_ret_frac, mv_vol_frac = portfolio_performance(min_vol.x, mean_returns, cov_matrix)
    mv_ret = mv_ret_frac * 100  # convert to %
    mv_vol = mv_vol_frac * 100  # convert to %
    mv_alloc = pd.DataFrame(min_vol.x, index=mean_returns.index, columns=['allocation'])
    mv_alloc.allocation = [round(i*100, 2) for i in mv_alloc.allocation]
    
    # Efficient Frontier (using target returns from min variance to 120% of max sharpe)
    target_returns_frac = np.linspace(mv_ret_frac, sr_ret_frac * 1.2, 50)
    efficient_frontier = []
    for ret in target_returns_frac:
        eff = efficient_optimization(mean_returns, cov_matrix, ret)
        eff_vol = np.sqrt(eff.fun) * 100  # convert to %
        efficient_frontier.append(eff_vol)
    target_returns = target_returns_frac * 100  # convert target returns to %
    
    return sr_ret, sr_vol, sr_alloc, mv_ret, mv_vol, mv_alloc, efficient_frontier, target_returns

# Create interactive efficient frontier plot with random portfolio scatter points
def plot_efficient_frontier(mean_returns, cov_matrix):
    sr_ret, sr_vol, sr_alloc, mv_ret, mv_vol, mv_alloc, ef_vols, target_rets = calculate_results(mean_returns, cov_matrix)
    
    # Generate random portfolios
    random_results = random_portfolios(mean_returns, cov_matrix)
    random_rets = random_results[0]
    random_vols = random_results[1]
    
    # Create traces
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
        x=random_vols,
        y=random_rets,
        mode='markers',
        marker=dict(size=4, color='rgba(0, 100, 255, 0.3)'),
        name='Random Portfolios'
    )
    
    # Create layout with improved aesthetics
    layout = go.Layout(
        title=dict(text='Portfolio Optimization with Modern Portfolio Theory',
                   x=0.5, xanchor='center', font=dict(size=24)),
        xaxis=dict(title='Annualized Volatility (%)', gridcolor='lightgray', zeroline=False, tickformat=".2f"),
        yaxis=dict(title='Annualized Returns (%)', gridcolor='lightgray', zeroline=False, tickformat=".2f"),
        legend=dict(x=0.75, y=0.15, bgcolor='rgba(255,255,255,0.7)'),
        width=1200,
        height=800,
        template='plotly_white'
    )
    
    # Create figure with all traces (random points behind the frontier curve)
    fig = go.Figure(data=[random_trace, ef_trace, max_sharpe_trace, min_vol_trace], layout=layout)
    fig.update_layout(xaxis_range=[min(random_vols)*0.9, max(random_vols)*1.1],
                      yaxis_range=[min(random_rets)*0.9, max(random_rets)*1.1])
    fig.show()

# Example usage
if __name__ == "__main__":
    stock_list = ['HDFCBANK.NS', 'INFY.NS', 'RELIANCE.NS'],
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=365)
    
    mean_returns, cov_matrix = get_data(stock_list, start_date, end_date)
    plot_efficient_frontier(mean_returns, cov_matrix)

