"""
risk_metrics.py
----------------
Historical-simulation Value at Risk (VaR) and Conditional Value at Risk
(CVaR / Expected Shortfall), plus drawdown and backtest utilities, applied
to a fixed-weight portfolio built from historical daily returns.
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def portfolio_daily_returns(returns: pd.DataFrame, weights: np.ndarray) -> pd.Series:
    return returns.values @ weights


def historical_var_cvar(port_daily_returns: np.ndarray, confidence: float = 0.95):
    """
    Historical-simulation 1-day VaR/CVaR at the given confidence level,
    expressed as a positive loss fraction (e.g. 0.023 = 2.3% loss).
    """
    losses = -port_daily_returns
    var = np.percentile(losses, confidence * 100)
    cvar = losses[losses >= var].mean()
    return var, cvar


def annualize_var_cvar(daily_var: float, daily_cvar: float, horizon_days: int = TRADING_DAYS):
    """Square-root-of-time scaling to approximate an annual VaR/CVaR from
    a 1-day historical VaR/CVaR (standard practical approximation)."""
    scale = np.sqrt(horizon_days)
    return daily_var * scale, daily_cvar * scale


def max_drawdown(cum_returns: pd.Series) -> float:
    running_max = cum_returns.cummax()
    drawdown = cum_returns / running_max - 1.0
    return drawdown.min()


def cumulative_growth(port_daily_returns: np.ndarray) -> np.ndarray:
    """Cumulative growth of $1 invested, from a series of daily simple
    returns (NOT log returns)."""
    simple_returns = np.expm1(port_daily_returns)
    return np.cumprod(1 + simple_returns)


def backtest_summary(returns: pd.DataFrame, weights: np.ndarray, label: str) -> dict:
    daily = portfolio_daily_returns(returns, weights)
    growth = cumulative_growth(daily)
    total_return = growth[-1] - 1
    var95, cvar95 = historical_var_cvar(daily, 0.95)
    ann_var95, ann_cvar95 = annualize_var_cvar(var95, cvar95)
    dd = max_drawdown(pd.Series(growth))

    return {
        "label": label,
        "total_return_pct": total_return * 100,
        "annual_return_pct": (np.expm1(daily.mean() * TRADING_DAYS)) * 100,
        "annual_vol_pct": (daily.std() * np.sqrt(TRADING_DAYS)) * 100,
        "daily_var_95_pct": var95 * 100,
        "daily_cvar_95_pct": cvar95 * 100,
        "annual_var_95_pct": ann_var95 * 100,
        "annual_cvar_95_pct": ann_cvar95 * 100,
        "max_drawdown_pct": dd * 100,
        "growth_curve": growth,
    }
