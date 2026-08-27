# Stochastic-Portfolio-Optimization-using-Monte-Carlo-Simulations
# Stochastic Portfolio Optimization using Monte Carlo Simulations

**Self Project | May 2025 – July 2025**

## 📌 Overview

This project focuses on **stochastic portfolio optimization of defense-sector equities** using large-scale **Monte Carlo simulations**. The objective was to identify an optimal portfolio allocation that maximizes expected returns while maintaining an acceptable level of downside risk.

A total of **10 million Monte Carlo simulations** were performed to evaluate a large number of possible portfolio allocations. Each simulated portfolio was assessed using multiple risk-adjusted performance measures, including:

* **Sharpe Ratio**
* **Value at Risk (VaR)**
* **Conditional Value at Risk (CVaR)**

The resulting optimized portfolio generated an approximate **150% return**, compared with approximately **125% for the HDFC Defence Fund** over the corresponding evaluation period.

---

## 🎯 Objective

The primary objective of this project was to develop a quantitative portfolio allocation strategy for **defense-sector equities** that could:

1. Maximize portfolio returns.
2. Improve risk-adjusted performance.
3. Control downside risk.
4. Identify an efficient allocation across selected defense-sector stocks.
5. Compare the optimized strategy against an existing defense-focused investment benchmark.

The project uses stochastic simulation rather than relying solely on historical mean-variance optimization, allowing a large number of potential portfolio outcomes to be explored.

---

## 🧠 Methodology

The overall methodology consisted of the following stages:

```text
Data Collection
      ↓
Data Cleaning & Preparation
      ↓
Historical Return Calculation
      ↓
Statistical Parameter Estimation
      ↓
Portfolio Weight Generation
      ↓
10 Million Monte Carlo Simulations
      ↓
Portfolio Performance Evaluation
      ↓
Risk Analysis
      ↓
Optimal Portfolio Selection
      ↓
Benchmark Comparison
```

---

## 📊 1. Data Collection

Historical market data for selected **defense-sector equities** was collected and prepared for quantitative analysis.

The dataset was used to calculate:

* Historical prices
* Daily returns
* Mean returns
* Volatility
* Correlation between securities
* Covariance structure

The analysis was performed using historical market observations to estimate the behavior of the selected securities.

> **Note:** The exact securities and data sources used in the analysis are provided in the project files/notebooks.

---

## 📈 2. Return Calculation

Daily percentage returns were calculated from historical closing prices.

For a security with price \(P_t\) at time \(t\), the simple daily return was calculated as:

$$
R_t = \frac{P_t-P_{t-1}}{P_{t-1}}
$$

Alternatively, logarithmic returns can be expressed as:

$$
r_t = \ln\left(\frac{P_t}{P_{t-1}}\right)
$$

The resulting return series was used to estimate the expected return and risk characteristics of individual securities.

---

## 📐 3. Portfolio Return

For a portfolio containing \(n\) assets, the portfolio return is calculated as:

$$
R_p = \sum_{i=1}^{n} w_iR_i
$$

where:

* \(R_p\) = portfolio return
* \(w_i\) = portfolio weight of asset \(i\)
* \(R_i\) = return of asset \(i\)

The portfolio weights satisfy:

$$
\sum_{i=1}^{n}w_i=1
$$

with non-negative weights used when implementing a long-only portfolio constraint.

---

## 🎲 4. Monte Carlo Simulation

The core of the project was a **Monte Carlo simulation framework**.

Instead of evaluating only a small number of predefined portfolios, millions of potential portfolio allocations were generated and evaluated.

### Simulation Process

For each simulation:

1. Generate a random set of portfolio weights.
2. Normalize the weights so that their sum equals 1.
3. Calculate the expected portfolio return.
4. Calculate portfolio volatility.
5. Calculate the Sharpe Ratio.
6. Estimate downside risk using VaR.
7. Estimate tail risk using CVaR.
8. Store the portfolio's performance metrics.
9. Repeat the process for **10 million simulations**.

A simplified representation is:

```python
for i in range(10_000_000):

    weights = generate_random_weights()
    weights = weights / weights.sum()

    portfolio_return = calculate_return(weights)
    portfolio_volatility = calculate_volatility(weights)

    sharpe_ratio = calculate_sharpe(
        portfolio_return,
        portfolio_volatility
    )

    VaR = calculate_var(portfolio_returns)
    CVaR = calculate_cvar(portfolio_returns)

    store_results(
        weights,
        portfolio_return,
        portfolio_volatility,
        sharpe_ratio,
        VaR,
        CVaR
    )
```

The simulation framework enabled the exploration of a very large portfolio search space.

---

# ⚖️ Risk Metrics

## 1. Sharpe Ratio

The Sharpe Ratio measures portfolio performance relative to the amount of risk taken.

$$
Sharpe=\frac{R_p-R_f}{\sigma_p}
$$

where:

* \(R_p\) = portfolio return
* \(R_f\) = risk-free rate
* \(\sigma_p\) = portfolio volatility

A higher Sharpe Ratio indicates better risk-adjusted performance.

The Sharpe Ratio was used as one of the primary criteria for identifying attractive portfolios.

---

## 2. Value at Risk (VaR)

**Value at Risk (VaR)** estimates the potential loss of a portfolio at a specified confidence level over a given time horizon.

For example, a 95% VaR represents a loss threshold that the portfolio is expected to exceed only approximately 5% of the time under the modeled distribution.

Conceptually:

$$
VaR_{\alpha}=-Q_{\alpha}(R)
$$

where \(Q_{\alpha}\) represents the relevant return percentile.

VaR was incorporated to evaluate the downside exposure of candidate portfolios.

---

## 3. Conditional Value at Risk (CVaR)

**Conditional Value at Risk (CVaR)**, also known as Expected Shortfall, measures the average loss in the worst tail of the return distribution.

Unlike VaR, which provides a loss threshold, CVaR evaluates the expected loss **beyond that threshold**.

$$
CVaR_{\alpha}=E[L\mid L\geq VaR_{\alpha}]
$$

CVaR therefore provides additional insight into the severity of extreme downside outcomes.

---

# 🔬 Portfolio Optimization Framework

The simulation results were used to identify portfolios offering attractive combinations of:

* Expected return
* Volatility
* Sharpe Ratio
* VaR
* CVaR

Rather than selecting a portfolio based purely on maximum return, the optimization considered the trade-off between **return generation and downside risk**.

A conceptual optimization objective can be represented as:

$$
\max \; f(R_p,\ Sharpe,\ VaR,\ CVaR)
$$

subject to portfolio allocation constraints such as:

$$
\sum_{i=1}^{n}w_i=1
$$

and:

$$
w_i\geq0
$$

The final portfolio was selected based on its overall risk-return characteristics.

---

# 📊 Results

The optimized strategy produced an approximate return of:

### **~150%**

This was compared with the performance of:

### **HDFC Defence Fund: ~125%**

| Metric          |      Optimized Portfolio | HDFC Defence Fund |
| --------------- | -----------------------: | ----------------: |
| Approx. Return  |                 **150%** |          **125%** |
| Approach        | Monte Carlo Optimization |    Fund Portfolio |
| Allocation      | Quantitatively Optimized |      Fund Managed |
| Risk Evaluation |        Sharpe, VaR, CVaR |         Benchmark |
| Simulation Runs |           **10 Million** |               N/A |

The optimized portfolio demonstrated an approximate **25 percentage-point return advantage** over the selected benchmark over the evaluation period.

> **Important:** The reported performance is based on the project's historical/backtested analysis and should not be interpreted as a guarantee of future investment performance.

---

# 📈 Key Findings

### 1. Large-scale simulation improves portfolio exploration

Running **10 million simulations** allowed the analysis to evaluate a very large number of possible portfolio allocations rather than relying on a limited set of manually selected combinations.

### 2. Return alone is insufficient

A portfolio with the highest expected return may also have significantly higher volatility or downside exposure.

Therefore, the analysis incorporated multiple risk metrics rather than optimizing solely for returns.

### 3. Risk-adjusted evaluation provides better insight

The **Sharpe Ratio** helped evaluate whether additional returns were justified by the associated volatility.

### 4. Tail-risk measures are important

VaR and CVaR provided additional information about potential downside outcomes, particularly during adverse market conditions.

### 5. Quantitative allocation can improve benchmark performance

The optimized portfolio generated approximately **150% returns**, compared with approximately **125% for the HDFC Defence Fund**, highlighting the potential benefit of systematic quantitative allocation.

---

# 🗂️ Project Structure

A suggested repository structure is:

```text
Stochastic-Portfolio-Optimization/
│
├── data/
│   ├── raw/
│   │   └── historical_data.csv
│   │
│   └── processed/
│       └── processed_returns.csv
│
├── notebooks/
│   ├── 01_Data_Collection.ipynb
│   ├── 02_Data_Analysis.ipynb
│   ├── 03_Monte_Carlo_Simulation.ipynb
│   └── 04_Portfolio_Optimization.ipynb
│
├── src/
│   ├── data_processing.py
│   ├── portfolio_metrics.py
│   ├── monte_carlo.py
│   └── optimization.py
│
├── results/
│   ├── portfolio_results.csv
│   ├── optimized_weights.csv
│   └── performance_comparison.png
│
├── requirements.txt
│
└── README.md
```

---

# 🛠️ Technologies Used

| Technology                 | Purpose                                   |
| -------------------------- | ----------------------------------------- |
| **Python**                 | Core programming language                 |
| **Pandas**                 | Data manipulation and analysis            |
| **NumPy**                  | Numerical computation                     |
| **Matplotlib**             | Data visualization                        |
| **Seaborn**                | Statistical visualization                 |
| **SciPy**                  | Optimization and statistical calculations |
| **Jupyter Notebook**       | Interactive analysis                      |
| **Monte Carlo Simulation** | Portfolio allocation search               |
| **Statistical Analysis**   | Risk and return estimation                |

---

# 💻 Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/<repository-name>.git
```

Navigate to the project directory:

```bash
cd Stochastic-Portfolio-Optimization
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Launch Jupyter Notebook:

```bash
jupyter notebook
```

Open the notebooks inside the `notebooks/` directory to reproduce the analysis.

---

# 📦 Requirements

Example `requirements.txt`:

```text
numpy
pandas
matplotlib
seaborn
scipy
jupyter
```

If market data was obtained using an API/library such as `yfinance`, add:

```text
yfinance
```

---

# 📉 Visualizations

The project can generate visualizations such as:

* Simulated portfolio return vs. volatility
* Efficient portfolio frontier
* Sharpe Ratio distribution
* Portfolio weight distribution
* VaR/CVaR comparison
* Optimized portfolio vs. benchmark performance
* Individual stock performance
* Correlation heatmap

Example conceptual visualization:

```text
                    Portfolio Return
                          ↑
                          │             ● Optimal Portfolio
                          │          ●
                          │       ●
                          │    ●
                          │  ●
                          │ ●
                          └────────────────────────→
                              Portfolio Risk
```

---

# 🔍 Reproducibility

The analysis can be reproduced by:

1. Obtaining the required historical market data.
2. Running the data preprocessing notebook.
3. Calculating historical returns and covariance.
4. Running the Monte Carlo simulation.
5. Evaluating portfolio-level performance metrics.
6. Selecting the optimized portfolio.
7. Comparing its historical performance against the benchmark.

Because Monte Carlo simulations use randomly generated portfolio weights, results can vary slightly between runs unless a fixed random seed is used.

For reproducible simulations, a random seed can be specified:

```python
np.random.seed(42)
```

---

# ⚠️ Limitations

The analysis has several limitations:

* Historical returns may not represent future market behavior.
* Monte Carlo simulations depend on assumptions about the underlying return distribution.
* Transaction costs and taxes may not be fully incorporated.
* Liquidity constraints may affect real-world execution.
* Extreme market events may not be adequately represented by historical data.
* The benchmark comparison depends on the selected evaluation period.
* Simulated optimal allocations may change as market conditions change.

Therefore, the results should be interpreted as a **quantitative research/backtesting exercise rather than investment advice**.

---

# 🚀 Future Improvements

Several extensions could make the model more robust:

### 1. Dynamic Portfolio Rebalancing

Implement periodic rebalancing based on changing market conditions.

### 2. Transaction Costs

Incorporate brokerage, slippage, taxes, and other transaction costs.

### 3. Alternative Optimization Objectives

Compare:

* Maximum Sharpe Ratio
* Minimum Volatility
* Minimum CVaR
* Maximum Return
* Risk-adjusted utility optimization

### 4. Machine Learning

Use machine-learning models to forecast:

* Returns
* Volatility
* Downside risk
* Market regimes

### 5. Regime-Based Optimization

Develop different allocation strategies for:

* Bull markets
* Bear markets
* High-volatility markets
* Low-volatility markets

### 6. Stress Testing

Evaluate portfolio performance under extreme historical scenarios and hypothetical market shocks.

---

# 📚 Key Concepts

This project demonstrates practical application of:

* Portfolio Theory
* Stochastic Optimization
* Monte Carlo Simulation
* Quantitative Finance
* Risk Management
* Statistical Analysis
* Value at Risk
* Conditional Value at Risk
* Sharpe Ratio
* Portfolio Diversification
* Backtesting

---

# 👤 Author

**Hari Sen**

B.Tech / Engineering Student

Interested in:

* Quantitative Finance
* Data Analytics
* Machine Learning
* Portfolio Optimization
* Financial Risk Management
* Product Management

---

# ⭐ Disclaimer

This project is intended for **educational and research purposes only**. The historical/backtested results presented in this repository do not constitute financial advice or a recommendation to buy or sell any security. Past performance does not guarantee future results.

---

## ⭐ If you find this project useful

Feel free to **star ⭐ the repository** and explore the notebooks to understand the complete Monte Carlo portfolio optimization workflow.
