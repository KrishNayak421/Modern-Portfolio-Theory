import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import scipy.optimize as sco

def portfolio_performance(weights, mean_returns, cov_matrix):
    """
    Calculate annualized portfolio return and volatility.
    """
    annual_return = np.dot(mean_returns, weights) * 252
    portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights)) * 252
    portfolio_volatility = np.sqrt(portfolio_variance)
    return annual_return, portfolio_volatility

def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate=0):
    """
    Returns the negative Sharpe ratio given weights, so that we can maximize it.
    """
    ret, vol = portfolio_performance(weights, mean_returns, cov_matrix)
    return -(ret - risk_free_rate) / vol

def max_sharpe_ratio(mean_returns, cov_matrix, risk_free_rate=0):
    """
    Optimizes for the maximum Sharpe ratio portfolio.
    """
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    result = sco.minimize(negative_sharpe_ratio,
                          num_assets * [1. / num_assets],
                          args=args,
                          method='SLSQP',
                          bounds=bounds,
                          constraints=constraints)
    return result

def portfolio_variance(weights, mean_returns, cov_matrix):
    """
    Returns the portfolio variance.
    """
    return np.dot(weights.T, np.dot(cov_matrix, weights)) * 252

def min_variance(mean_returns, cov_matrix):
    """
    Optimizes for the minimum volatility portfolio.
    """
    num_assets = len(mean_returns)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(num_assets))
    
    result = sco.minimize(portfolio_variance,
                          num_assets * [1. / num_assets],
                          args=(mean_returns, cov_matrix),
                          method='SLSQP',
                          bounds=bounds,
                          constraints=constraints)
    return result

def main():
    # ------------------ User Input ------------------ #
    # Prompt the user for stock tickers and weights.
    stocks_input = input("Enter stock tickers separated by commas (e.g., AAPL,MSFT,GOOGL): ")
    stock_list = [ticker.strip().upper() for ticker in stocks_input.split(',') if ticker.strip() != ""]
    
    weights_input = input("Enter corresponding weights in percentages separated by commas (e.g., 40,30,30): ")
    try:
        weights = [float(w.strip()) / 100 for w in weights_input.split(',') if w.strip() != ""]
    except ValueError:
        print("Invalid weights entered. Please enter numeric values.")
        return
    
    if len(stock_list) != len(weights):
        print("The number of stocks and weights must be the same.")
        return
    
    if not np.isclose(sum(weights), 1):
        print("The weights do not sum to 100%. Please check your input.")
        return

    # ------------------ Data Download ------------------ #
    # Set a historical period (e.g., 5 years).
    end_date = dt.datetime.today()
    start_date = end_date - dt.timedelta(days=365 * 5)
    
    print("\nDownloading historical data...")
    data = yf.download(stock_list, start=start_date, end=end_date, auto_adjust=True)['Close']
    data.dropna(inplace=True)  # Ensure no missing values
    
    # Compute daily returns
    daily_returns = data.pct_change().dropna()
    
    # Calculate mean daily returns and covariance matrix.
    mean_returns = daily_returns.mean()
    cov_matrix = daily_returns.cov()
    
    # ------------------ Portfolio Performance for Input Weights ------------------ #
    port_ret, port_vol = portfolio_performance(np.array(weights), mean_returns, cov_matrix)
    sharpe_ratio = (port_ret / port_vol) if port_vol != 0 else 0
    
    print("\nProvided Portfolio Performance:")
    print(f"Annualized Return    : {port_ret * 100:.2f}%")
    print(f"Annualized Volatility: {port_vol * 100:.2f}%")
    print(f"Sharpe Ratio         : {sharpe_ratio:.2f}")
    
    # ------------------ Maximum Sharpe Ratio Portfolio ------------------ #
    max_sharpe = max_sharpe_ratio(mean_returns, cov_matrix)
    max_sharpe_weights = max_sharpe.x
    ms_ret, ms_vol = portfolio_performance(max_sharpe_weights, mean_returns, cov_matrix)
    ms_sharpe = (ms_ret / ms_vol) if ms_vol != 0 else 0
    
    print("\nMaximum Sharpe Ratio Portfolio:")
    for stock, weight in zip(stock_list, max_sharpe_weights):
        print(f"  {stock}: {weight * 100:.2f}%")
    print(f"Annualized Return    : {ms_ret * 100:.2f}%")
    print(f"Annualized Volatility: {ms_vol * 100:.2f}%")
    print(f"Sharpe Ratio         : {ms_sharpe:.2f}")
    
    # ------------------ Minimum Volatility Portfolio ------------------ #
    min_vol = min_variance(mean_returns, cov_matrix)
    min_vol_weights = min_vol.x
    mv_ret, mv_vol = portfolio_performance(min_vol_weights, mean_returns, cov_matrix)
    mv_sharpe = (mv_ret / mv_vol) if mv_vol != 0 else 0
    
    print("\nMinimum Volatility Portfolio:")
    for stock, weight in zip(stock_list, min_vol_weights):
        print(f"  {stock}: {weight * 100:.2f}%")
    print(f"Annualized Return    : {mv_ret * 100:.2f}%")
    print(f"Annualized Volatility: {mv_vol * 100:.2f}%")
    print(f"Sharpe Ratio         : {mv_sharpe:.2f}")

if __name__ == '__main__':
    main()
