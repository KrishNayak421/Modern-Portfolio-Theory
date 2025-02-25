from eod import EODHistoricalData

# Initialize the client with your API key
client = EODHistoricalData('YOUR_API_KEY')

# Retrieve historical prices for a bond
bond_symbol = 'SW10Y.GBOND'
bond_prices = client.get_prices_eod(bond_symbol, period='d', order='a')

# Convert the data to a DataFrame
df = pd.DataFrame(bond_prices)
df.set_index('date', inplace=True)

print(df)
