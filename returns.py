import yfinance as yf
import pandas as pd
import datetime as dt
import plotly.express as px

def get_annualized_returns(stock_list, start, end):
    """
    Fetch historical adjusted closing prices for the provided stocks and 
    calculate the annualized returns.
    
    Annualized Return = (Mean Daily Return) * 252
    
    Parameters:
    - stock_list: List of stock tickers (e.g., ["AAPL", "MSFT", "GOOGL"])
    - start: Start date (datetime object)
    - end: End date (datetime object)
    
    Returns:
    - annualized_returns: A pandas Series with stock tickers as the index
                          and annualized returns as values (in fraction form).
    """
    # Download historical adjusted closing prices
    data = yf.download(stock_list, start=start, end=end, auto_adjust=True)['Close']
    
    # Calculate daily returns and drop the first NaN row
    daily_returns = data.pct_change().dropna()
    
    # Calculate average daily return and then annualize it (252 trading days per year)
    daily_mean_returns = daily_returns.mean()
    annualized_returns = daily_mean_returns * 252
    
    return annualized_returns

def plot_annualized_returns(annualized_returns):
    """
    Create a bar chart using Plotly to visualize the annualized returns.
    
    Parameters:
    - annualized_returns: pandas Series with the annualized returns.
    """
    # Convert the Series to a DataFrame for Plotly Express
    df_returns = annualized_returns.reset_index()
    df_returns.columns = ['Stock', 'Annualized Return']
    
    # Create bar chart
    fig = px.bar(
        df_returns, 
        x='Stock', 
        y='Annualized Return', 
        title="Annualized Returns of Stocks",
        labels={'Annualized Return': 'Annualized Return (Fraction)'},
        text='Annualized Return'
    )
    
    # Update layout for better visualization
    fig.update_traces(texttemplate='%{text:.2%}', textposition='outside', marker_color='royalblue')
    fig.update_layout(yaxis_tickformat=',.2%', uniformtext_minsize=8, uniformtext_mode='hide')
    
    fig.show()

def main():
    # Define list of stocks (modify this list as needed)
    stock_list = [ "HDFCBANK.NS",
        "INFY.NS",
        "BAJFINANCE.NS",
       # "INDUSINDBK.NS",
        "BALRAMCHIN.NS",
        # "INDUSTOWER.NS",
        # "BANKBEES.NS",  
        "CPSEETF.NS",   
        "BANKETF.NS",    
        "SETFNIF50.NS", ]
    
    # Define the date range (e.g., last 5 years)
    end_date = dt.datetime.today()
    start_date = end_date - dt.timedelta(days=365*5)
    
    # Calculate annualized returns
    annualized_returns = get_annualized_returns(stock_list, start_date, end_date)
    
    # Print returns to the console
    print("Annualized Returns:")
    print(annualized_returns.apply(lambda x: f"{x:.2%}"))
    
    # Plot the annualized returns using Plotly
    plot_annualized_returns(annualized_returns)

if __name__ == '__main__':
    main()
