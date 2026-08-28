"""
data_loader.py
--------------
Loads daily adjusted-close price data for a basket of Indian defense-sector
equities, plus a benchmark series (HDFC Defence Fund NAV proxy).

Two modes:
  1. LIVE   - pulls real data from Yahoo Finance via yfinance (requires
              internet access). This is the recommended mode when running
              the project locally / in GitHub Actions.
  2. SYNTHETIC - if yfinance / internet is unavailable, falls back to a
              reproducible Geometric Brownian Motion (GBM) simulation whose
              drift/volatility parameters are calibrated to roughly match
              the real 2021-2024 rally seen in Indian defense stocks. This
              keeps the pipeline fully runnable offline (e.g. in CI, or in
              this sandboxed environment) while being clearly labelled as
              synthetic data.

Set FORCE_SYNTHETIC = True to always use synthetic data (useful for fast,
fully-reproducible demo runs).
"""

import numpy as np
import pandas as pd

FORCE_SYNTHETIC = True  # set to False to attempt a live yfinance pull first

# Defense-sector universe (NSE tickers)
TICKERS = {
    "HAL.NS": "Hindustan Aeronautics",
    "BEL.NS": "Bharat Electronics",
    "BDL.NS": "Bharat Dynamics",
    "MAZDOCK.NS": "Mazagon Dock Shipbuilders",
    "COCHINSHIP.NS": "Cochin Shipyard",
    "SOLARINDS.NS": "Solar Industries India",
}

BENCHMARK_NAME = "HDFC Defence Fund (proxy)"

START_DATE = "2021-01-01"
END_DATE = "2024-12-31"

# Calibration parameters for the synthetic fallback, roughly reflecting the
# real-world 2021-2024 rally + drawdowns in this sector.
# (annual_drift, annual_volatility)
SYNTHETIC_PARAMS = {
    "HAL.NS":        (0.26, 0.30),
    "BEL.NS":        (0.24, 0.32),
    "BDL.NS":        (0.28, 0.40),
    "MAZDOCK.NS":    (0.30, 0.48),
    "COCHINSHIP.NS": (0.25, 0.42),
    "SOLARINDS.NS":  (0.20, 0.34),
}
BENCHMARK_PARAMS = (0.2027, 0.16)  # calibrated so cumulative return ~125% over the 4Y window

SEED = 16


def _gbm_price_path(s0, mu, sigma, n_days, dt, rng):
    """Simulate one GBM daily price path."""
    shocks = rng.normal(
        loc=(mu - 0.5 * sigma ** 2) * dt,
        scale=sigma * np.sqrt(dt),
        size=n_days,
    )
    log_path = np.cumsum(shocks)
    return s0 * np.exp(log_path)


def _correlated_shocks(n_assets, n_days, dt, rng, base_corr=0.45):
    """Generate correlated daily standard-normal shocks (assets are driven
    by a shared sector factor + idiosyncratic noise, typical of stocks in
    the same industry)."""
    corr = np.full((n_assets, n_assets), base_corr)
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)
    z = rng.normal(size=(n_days, n_assets))
    return z @ chol.T


def generate_synthetic_data():
    """Builds a reproducible synthetic price DataFrame for the defense
    basket plus benchmark, using correlated GBM paths."""
    rng = np.random.default_rng(SEED)
    dates = pd.bdate_range(START_DATE, END_DATE)
    n_days = len(dates)
    dt = 1 / 252

    tickers = list(TICKERS.keys())
    n_assets = len(tickers)
    z = _correlated_shocks(n_assets, n_days, dt, rng)

    prices = {}
    for i, tkr in enumerate(tickers):
        mu, sigma = SYNTHETIC_PARAMS[tkr]
        drift = (mu - 0.5 * sigma ** 2) * dt
        diffusion = sigma * np.sqrt(dt) * z[:, i]
        log_returns = drift + diffusion
        price_path = 100 * np.exp(np.cumsum(log_returns))
        prices[tkr] = price_path

    # Benchmark: independent, lower-vol GBM path
    mu_b, sigma_b = BENCHMARK_PARAMS
    z_b = rng.normal(size=n_days)
    bench_returns = (mu_b - 0.5 * sigma_b ** 2) * dt + sigma_b * np.sqrt(dt) * z_b
    prices[BENCHMARK_NAME] = 100 * np.exp(np.cumsum(bench_returns))

    df = pd.DataFrame(prices, index=dates)
    df.index.name = "Date"
    return df


def load_price_data():
    """Attempts a live pull via yfinance; falls back to synthetic data.

    Returns
    -------
    prices : pd.DataFrame  (Date index, one column per ticker + benchmark)
    is_synthetic : bool
    """
    if not FORCE_SYNTHETIC:
        try:
            import yfinance as yf

            tickers = list(TICKERS.keys())
            raw = yf.download(tickers, start=START_DATE, end=END_DATE)["Adj Close"]
            raw = raw.dropna(how="all")
            if raw.empty:
                raise ValueError("empty download")
            # NOTE: a real benchmark NAV series (HDFC Defence Fund) would be
            # pulled from AMFI / mutual-fund NAV history here.
            return raw, False
        except Exception as exc:  # noqa: BLE001
            print(f"[data_loader] Live data fetch failed ({exc}); "
                  f"falling back to synthetic dataset.")

    return generate_synthetic_data(), True


if __name__ == "__main__":
    data, synth = load_price_data()
    print(f"Synthetic: {synth}")
    print(data.head())
    print(data.tail())
