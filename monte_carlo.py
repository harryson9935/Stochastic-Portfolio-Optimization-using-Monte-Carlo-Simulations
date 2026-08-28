"""
monte_carlo.py
---------------
Vectorized Monte Carlo engine for stochastic portfolio optimization.

For N_SIMULATIONS random long-only, fully-invested weight vectors, we
compute the resulting portfolio's annualized expected return, annualized
volatility, and Sharpe ratio using historical daily log returns of the
asset universe. The simulation is fully vectorized with NumPy so it scales
to millions of trials in seconds.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252
RISK_FREE_RATE = 0.07  # ~ Indian 10Y G-Sec / T-bill proxy, annualized


def daily_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return np.log(prices / prices.shift(1)).dropna()


def sample_random_weights(n_simulations: int, n_assets: int, rng: np.random.Generator) -> np.ndarray:
    """Long-only, fully-invested weights drawn uniformly on the simplex
    (via normalized exponential/Dirichlet-style sampling)."""
    raw = rng.exponential(scale=1.0, size=(n_simulations, n_assets))
    weights = raw / raw.sum(axis=1, keepdims=True)
    return weights


def run_monte_carlo(returns: pd.DataFrame, n_simulations: int = 1_000_000, seed: int = 7):
    """
    Runs the vectorized Monte Carlo simulation.

    Parameters
    ----------
    returns : pd.DataFrame of daily log returns (columns = assets)
    n_simulations : number of random portfolios to sample
    seed : RNG seed for reproducibility

    Returns
    -------
    dict with weights, exp_returns, volatilities, sharpe_ratios (all np.ndarray),
    plus the mean vector and covariance matrix used.
    """
    rng = np.random.default_rng(seed)
    n_assets = returns.shape[1]

    mean_daily = returns.mean().values
    cov_daily = returns.cov().values

    mean_annual = mean_daily * TRADING_DAYS
    cov_annual = cov_daily * TRADING_DAYS

    # Process in chunks to keep peak memory bounded for very large N
    chunk_size = 200_000
    all_weights = np.empty((n_simulations, n_assets), dtype=np.float64)
    exp_returns = np.empty(n_simulations, dtype=np.float64)
    volatilities = np.empty(n_simulations, dtype=np.float64)

    start = 0
    while start < n_simulations:
        end = min(start + chunk_size, n_simulations)
        n_chunk = end - start

        w = sample_random_weights(n_chunk, n_assets, rng)
        port_return = w @ mean_annual
        # portfolio variance = w^T Cov w, vectorized across all rows
        port_var = np.einsum("ij,jk,ik->i", w, cov_annual, w)
        port_vol = np.sqrt(port_var)

        all_weights[start:end] = w
        exp_returns[start:end] = port_return
        volatilities[start:end] = port_vol
        start = end

    sharpe_ratios = (exp_returns - RISK_FREE_RATE) / volatilities

    return {
        "weights": all_weights,
        "exp_returns": exp_returns,
        "volatilities": volatilities,
        "sharpe_ratios": sharpe_ratios,
        "mean_annual": mean_annual,
        "cov_annual": cov_annual,
        "assets": list(returns.columns),
    }


def best_portfolios(mc_result: dict):
    """Extract the max-Sharpe and min-volatility portfolios from a MC run."""
    sharpe = mc_result["sharpe_ratios"]
    vol = mc_result["volatilities"]

    max_sharpe_idx = int(np.argmax(sharpe))
    min_vol_idx = int(np.argmin(vol))

    def _pack(idx):
        return {
            "weights": mc_result["weights"][idx],
            "exp_return": mc_result["exp_returns"][idx],
            "volatility": mc_result["volatilities"][idx],
            "sharpe": mc_result["sharpe_ratios"][idx],
        }

    return {
        "max_sharpe": _pack(max_sharpe_idx),
        "min_volatility": _pack(min_vol_idx),
    }
