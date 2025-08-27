import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

def download_data(tickers, start_date, end_date):
    """
    Downloads historical adjusted closing prices for the given tickers.
    """
    data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']
    return data

def plot_correlation_heatmap(data):
    """
    Computes daily returns, calculates the correlation matrix, and plots a heatmap.
    """
    # Calculate daily returns 
    returns = data.pct_change().dropna()
    corr_matrix = returns.corr()

    # heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap="Blues", fmt=".2f", linewidths=0.5)
    plt.title("Stock Returns Correlation Matrix")
    plt.show()


def main():
    # Hardcoded list of stock tickers
    tickers = [
        "HDFCBANK.NS",
        "INFY.NS",
        "BALRAMCHIN.NS",
        "INDUSTOWER.NS",
        "CIPLA.NS",
        "KPIGREEN.NS",
        "DHANBANK.NS",
    ]
    
    # date range (5 yrs)
    end_date = dt.datetime.now()
    start_date = end_date - dt.timedelta(days=365*5)
    
    data = download_data(tickers, start_date, end_date)
    plot_correlation_heatmap(data)

if __name__ == "__main__":
    main()
