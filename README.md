# **Modern Portfolio Theory (MPT)**

**Modern Portfolio Theory (MPT)**, developed by Harry Markowitz, is a mathematical framework for constructing an optimal investment portfolio. MPT focuses on maximizing returns for a given level of risk or minimizing risk for a given expected return through diversification.

---

## **Key Concepts**

1. **Expected Return:**  
   The anticipated return of a portfolio based on the weighted average of individual asset returns.

   $$
   E(R_p) = \sum_{i=1}^{n} w_i E(R_i)
   $$

2. **Portfolio Risk (Variance and Covariance):**  
   The overall risk of the portfolio is affected by the variance of each asset and the covariance between assets:

   $$
   \sigma_p^2 = \sum_{i=1}^{n} \sum_{j=1}^{n} w_i w_j \text{Cov}(R_i, R_j)
   $$

3. **Efficient Frontier:**  
   A set of portfolios offering the highest possible return for each level of risk. Portfolios on this curve are considered optimal.

---

## **How It Works**

- MPT aims to reduce overall portfolio risk through diversification.
- It constructs portfolios by varying asset weights to find the best tradeoff between risk and return.

---

## **Implementation**

Here’s a simple Python implementation to calculate the expected return and risk of a portfolio:

```python
import numpy as np

# Step 1: Define expected returns and covariance matrix
expected_returns = np.array([0.08, 0.12, 0.10])  # Expected returns of three assets
cov_matrix = np.array([
    [0.04, 0.02, 0.01],
    [0.02, 0.03, 0.015],
    [0.01, 0.015, 0.02]
])  # Covariance matrix of asset returns
weights = np.array([0.5, 0.3, 0.2])  # Portfolio weights (sum should equal 1)

# Step 2: Calculate portfolio expected return
portfolio_return = np.dot(weights, expected_returns)

# Step 3: Calculate portfolio variance
portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))

# Step 4: Calculate portfolio risk (standard deviation)
portfolio_std_dev = np.sqrt(portfolio_variance)

# Step 5: Output the results
print("Portfolio Expected Return:", portfolio_return)
print("Portfolio Risk (Standard Deviation):", portfolio_std_dev)
```

---

## **Advantages of MPT**

1. **Diversification:** Reduces risk through asset allocation.
2. **Risk-Return Optimization:** Helps create a balance between risk and reward.
3. **Efficient Portfolios:** Identifies the best possible portfolios for each risk level.

---

## **Limitations of MPT**

1. **Assumes Normal Distribution:** MPT assumes returns are normally distributed, which may not reflect real-world events.
2. **Estimation Sensitivity:** Errors in estimating expected returns or covariances can lead to suboptimal portfolios.
3. **Static Model:** MPT does not account for dynamic changes in market conditions.

---



