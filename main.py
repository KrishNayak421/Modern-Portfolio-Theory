# Install dependencies
import time 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import datetime as dt
import scipy as sc

# Import data
def getData(stocks, start, end):
    stockData = yf.download(stocks, start=start, end=end, auto_adjust=True)
    stockData = stockData['Close']
    returns = stockData.pct_change(fill_method=None)
    mean_returns = returns.mean()
    cov_matrix = returns.cov()
    return mean_returns, cov_matrix

def PortfolioPerformance(weights, MeanReturns, covMatrix):
    returns = np.sum(MeanReturns * weights) * 252
    std = np.sqrt(np.dot(weights.T, np.dot(covMatrix, weights))) * np.sqrt(252)
    return returns, std    

def negative_sharpe_ratio(weights, MeanReturns, covMatrix, riskFreeRate = 0):
    p_ret, p_std = PortfolioPerformance(weights, MeanReturns, covMatrix)
    return -(p_ret - riskFreeRate) / p_std


def maxSR(meanReturns, covMatrix, riskFreeRate = 0, constraintset = (0,1)):
    "Minimize -ve sharpe ratio"
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix, riskFreeRate)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constraintset
    bounds = tuple(bound for asset in range(numAssets))
    result = sc.optimize.minimize(negative_sharpe_ratio, numAssets*[1./numAssets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def portfolioVariance(weights, meanReturns, covMatrix):
    return PortfolioPerformance(weights, meanReturns, covMatrix)[1]

def minimizeVariance(meanReturns, covMatrix, constrainSet=(0,1)):
    numAssets = len(meanReturns)
    args = (meanReturns, covMatrix)
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bound = constrainSet
    bounds = tuple(bound for asset in range(numAssets))
    result = sc.optimize.minimize(portfolioVariance, numAssets*[1./numAssets,], args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result



#weights = np.array([0.33, 0.33, 0.33])
stockList = ['HDFCBANK.NS', 'INFY.NS','RELIANCE.NS']
endtime = dt.datetime.now()
starttime = endtime - dt.timedelta(days=365)

mean_returns, cov_matrix = getData(stockList, starttime, endtime)
#returns, std = PortfolioPerformance(weights, mean_returns, cov_matrix)
# results = maxSR(mean_returns, cov_matrix)
# maxSR , maxWeights = results['fun'],results['x']
# print(maxSR, maxWeights)


# minVarResult = minimizeVariance(mean_returns, cov_matrix)
# minVar, minVarWeights = minVarResult['fun'], minVarResult['x']
# print(minVar, minVarWeights)


def CalculatedResults(meanReturns, covMatrix, riskFreeRate = 0, constrainset = (0,1)):
    maxSR_Portfolio = maxSR(mean_returns, cov_matrix)
    maxSR_returns, maxSR_std = PortfolioPerformance( maxSR_Portfolio['x'], meanReturns, covMatrix)
    maxSR_allocation = pd.DataFrame(maxSR_Portfolio['x'], index=meanReturns.index, columns=['allocation'])
    maxSR_allocation.allocation = [round(i*100,2) for i in maxSR_allocation.allocation]
    

    minVar_Portfolio = minimizeVariance(mean_returns, cov_matrix)
    minVal_returns, minVar_std = PortfolioPerformance(minVar_Portfolio['x'], meanReturns, covMatrix)
    minVal_allocation = pd.DataFrame(minVar_Portfolio['x'], index=meanReturns.index, columns=['allocation'])
    minVal_allocation.allocation = [round(i*100,2) for i in minVal_allocation.allocation]
    return minVal_returns, minVar_std, minVal_allocation


print(CalculatedResults(mean_returns, cov_matrix))