"""
main.py
-------
End-to-end pipeline for the Stochastic Portfolio Optimization project.

Run:
    python src/main.py

Produces (in ../results/):
    efficient_frontier.png
    portfolio_allocation.png
    cumulative_returns.png
    risk_metrics_comparison.png
    correlation_heatmap.png
    results_summary.csv
    weights_max_sharpe.csv
"""

import os
import sys
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

from data_loader import load_price_data, TICKERS, BENCHMARK_NAME
from monte_carlo import daily_log_returns, run_monte_carlo, best_portfolios, RISK_FREE_RATE, TRADING_DAYS
from risk_metrics import backtest_summary

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

N_SIMULATIONS = 1_000_000
# NOTE: the original project ran 10,000,000 simulations. The vectorized
# engine in monte_carlo.py handles that easily on a laptop (~15-25s);
# N_SIMULATIONS is set lower here to keep the default demo run fast in
# constrained CI/sandbox environments. Bump it back to 10_000_000 to
# reproduce the full-scale run.

plt.style.use("seaborn-v0_8-darkgrid")
COLORS = {
    "primary": "#0b3d2e",
    "accent": "#c9a227",
    "highlight": "#a4243b",
    "neutral": "#4a5859",
}


def short_name(ticker: str) -> str:
    return ticker.replace(".NS", "")


def main():
    t0 = time.time()
    print("=" * 70)
    print("STOCHASTIC PORTFOLIO OPTIMIZATION USING MONTE CARLO SIMULATIONS")
    print("=" * 70)

    # ---------------------------------------------------------------
    # 1. Load data
    # ---------------------------------------------------------------
    prices, is_synthetic = load_price_data()
    asset_cols = [c for c in prices.columns if c != BENCHMARK_NAME]
    print(f"\n[1/6] Data loaded ({'SYNTHETIC (offline demo)' if is_synthetic else 'LIVE'})")
    print(f"      Universe: {', '.join(asset_cols)}")
    print(f"      Period  : {prices.index.min().date()} -> {prices.index.max().date()}  "
          f"({len(prices)} trading days)")

    asset_returns = daily_log_returns(prices[asset_cols])
    bench_returns = daily_log_returns(prices[[BENCHMARK_NAME]])

    # ---------------------------------------------------------------
    # 2. Monte Carlo simulation
    # ---------------------------------------------------------------
    print(f"\n[2/6] Running {N_SIMULATIONS:,} Monte Carlo simulations ...")
    t1 = time.time()
    mc = run_monte_carlo(asset_returns, n_simulations=N_SIMULATIONS, seed=7)
    print(f"      Done in {time.time() - t1:.2f}s")

    best = best_portfolios(mc)
    max_sharpe = best["max_sharpe"]
    min_vol = best["min_volatility"]

    print(f"\n      Max-Sharpe portfolio  -> Return: {max_sharpe['exp_return']*100:6.2f}%   "
          f"Vol: {max_sharpe['volatility']*100:6.2f}%   Sharpe: {max_sharpe['sharpe']:.3f}")
    print(f"      Min-Vol portfolio     -> Return: {min_vol['exp_return']*100:6.2f}%   "
          f"Vol: {min_vol['volatility']*100:6.2f}%   Sharpe: {min_vol['sharpe']:.3f}")

    # ---------------------------------------------------------------
    # 3. Backtest optimal portfolio vs benchmark
    # ---------------------------------------------------------------
    print("\n[3/6] Backtesting optimal (max-Sharpe) portfolio vs benchmark ...")
    opt_summary = backtest_summary(asset_returns, max_sharpe["weights"], "Optimized Portfolio (Max Sharpe)")
    bench_weights = np.array([1.0])
    bench_summary = backtest_summary(bench_returns, bench_weights, BENCHMARK_NAME)

    print(f"      Optimized portfolio total return : {opt_summary['total_return_pct']:.1f}%")
    print(f"      {BENCHMARK_NAME} total return      : {bench_summary['total_return_pct']:.1f}%")

    # ---------------------------------------------------------------
    # 4. Plots
    # ---------------------------------------------------------------
    print("\n[4/6] Generating plots ...")
    plot_efficient_frontier(mc, max_sharpe, min_vol)
    plot_allocation(max_sharpe["weights"], asset_cols)
    plot_cumulative_returns(opt_summary["growth_curve"], bench_summary["growth_curve"], prices.index[1:])
    plot_risk_metrics(opt_summary, bench_summary)
    plot_correlation_heatmap(asset_returns, asset_cols)

    # ---------------------------------------------------------------
    # 5. Save numeric results
    # ---------------------------------------------------------------
    print("\n[5/6] Saving results tables ...")
    save_results_tables(mc, max_sharpe, min_vol, opt_summary, bench_summary, asset_cols)

    # ---------------------------------------------------------------
    # 6. Console summary
    # ---------------------------------------------------------------
    print("\n[6/6] Summary")
    print("-" * 70)
    outperformance = opt_summary["total_return_pct"] - bench_summary["total_return_pct"]
    print(f"      Optimized strategy outperformed benchmark by {outperformance:+.1f} percentage points")
    print(f"      Total runtime: {time.time() - t0:.2f}s")
    print("=" * 70)


def plot_efficient_frontier(mc, max_sharpe, min_vol):
    # Subsample for a readable scatter (plotting 1M points is wasteful)
    n = len(mc["sharpe_ratios"])
    idx = np.random.default_rng(1).choice(n, size=min(30_000, n), replace=False)

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(
        mc["volatilities"][idx] * 100,
        mc["exp_returns"][idx] * 100,
        c=mc["sharpe_ratios"][idx],
        cmap="viridis",
        s=6,
        alpha=0.5,
    )
    cbar = plt.colorbar(sc)
    cbar.set_label("Sharpe Ratio")

    ax.scatter(max_sharpe["volatility"] * 100, max_sharpe["exp_return"] * 100,
               marker="*", color=COLORS["highlight"], s=500, edgecolor="black",
               linewidth=1, label="Max Sharpe Portfolio", zorder=5)
    ax.scatter(min_vol["volatility"] * 100, min_vol["exp_return"] * 100,
               marker="D", color=COLORS["accent"], s=140, edgecolor="black",
               linewidth=1, label="Min Volatility Portfolio", zorder=5)

    ax.set_xlabel("Annualized Volatility (%)")
    ax.set_ylabel("Annualized Expected Return (%)")
    ax.set_title(f"Efficient Frontier — {len(mc['sharpe_ratios']):,} Monte Carlo Simulated Portfolios")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "efficient_frontier.png"), dpi=160)
    plt.close(fig)


def plot_allocation(weights, asset_cols):
    labels = [short_name(a) for a in asset_cols]
    legend_labels = [f"{lbl} — {TICKERS.get(a, a)}" for lbl, a in zip(labels, asset_cols)]

    fig, ax = plt.subplots(figsize=(11, 7))
    palette = plt.cm.tab20c(np.linspace(0, 1, len(weights)))

    def autopct_fmt(pct):
        return f"{pct:.1f}%" if pct >= 2.0 else ""

    wedges, texts, autotexts = ax.pie(
        weights * 100,
        labels=None,
        autopct=autopct_fmt,
        startangle=90,
        colors=palette,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        pctdistance=0.78,
    )
    for t in autotexts:
        t.set_fontsize(10)
        t.set_color("white")
        t.set_fontweight("bold")

    ax.legend(wedges, legend_labels, title="Asset", loc="center left",
              bbox_to_anchor=(0.98, 0.5), fontsize=9, frameon=False)
    ax.set_title("Optimal Portfolio Allocation (Max Sharpe Ratio)")
    fig.subplots_adjust(right=0.62)
    fig.savefig(os.path.join(RESULTS_DIR, "portfolio_allocation.png"), dpi=160)
    plt.close(fig)


def plot_cumulative_returns(opt_growth, bench_growth, dates):
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(dates, (opt_growth - 1) * 100, label="Optimized Portfolio (Max Sharpe)",
            color=COLORS["primary"], linewidth=2)
    ax.plot(dates, (bench_growth - 1) * 100, label="HDFC Defence Fund (proxy)",
            color=COLORS["highlight"], linewidth=2, linestyle="--")
    ax.axhline(0, color="grey", linewidth=0.8)
    ax.set_ylabel("Cumulative Return (%)")
    ax.set_title("Cumulative Return: Optimized Portfolio vs Benchmark")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "cumulative_returns.png"), dpi=160)
    plt.close(fig)


def plot_risk_metrics(opt_summary, bench_summary):
    metrics = ["annual_return_pct", "annual_vol_pct", "annual_var_95_pct", "annual_cvar_95_pct", "max_drawdown_pct"]
    metric_labels = ["Annual\nReturn", "Annual\nVolatility", "Annual VaR\n(95%)", "Annual CVaR\n(95%)", "Max\nDrawdown"]

    opt_vals = [abs(opt_summary[m]) for m in metrics]
    bench_vals = [abs(bench_summary[m]) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(x - width / 2, opt_vals, width, label="Optimized Portfolio", color=COLORS["primary"])
    ax.bar(x + width / 2, bench_vals, width, label="Benchmark (HDFC Defence Fund proxy)", color=COLORS["highlight"])
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("% (absolute value)")
    ax.set_title("Risk & Return Metrics: Optimized Portfolio vs Benchmark")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "risk_metrics_comparison.png"), dpi=160)
    plt.close(fig)


def plot_correlation_heatmap(returns, asset_cols):
    corr = returns.corr()
    labels = [short_name(a) for a in asset_cols]

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(corr.values, cmap="RdYlGn_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                     color="black", fontsize=9)
    ax.set_title("Asset Correlation Matrix (Daily Log Returns)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "correlation_heatmap.png"), dpi=160)
    plt.close(fig)


def save_results_tables(mc, max_sharpe, min_vol, opt_summary, bench_summary, asset_cols):
    weights_df = pd.DataFrame({
        "Asset": asset_cols,
        "Company": [TICKERS.get(a, a) for a in asset_cols],
        "Max_Sharpe_Weight_%": np.round(max_sharpe["weights"] * 100, 2),
        "Min_Vol_Weight_%": np.round(min_vol["weights"] * 100, 2),
    })
    weights_df.to_csv(os.path.join(RESULTS_DIR, "weights_max_sharpe.csv"), index=False)

    summary_rows = [
        {
            "Metric": "Number of Simulations",
            "Optimized Portfolio (Max Sharpe)": f"{len(mc['sharpe_ratios']):,}",
            "Benchmark": "-",
        },
        {
            "Metric": "Expected Annual Return (from MC)",
            "Optimized Portfolio (Max Sharpe)": f"{max_sharpe['exp_return']*100:.2f}%",
            "Benchmark": "-",
        },
        {
            "Metric": "Expected Annual Volatility (from MC)",
            "Optimized Portfolio (Max Sharpe)": f"{max_sharpe['volatility']*100:.2f}%",
            "Benchmark": "-",
        },
        {
            "Metric": "Sharpe Ratio (from MC)",
            "Optimized Portfolio (Max Sharpe)": f"{max_sharpe['sharpe']:.3f}",
            "Benchmark": "-",
        },
        {
            "Metric": "Backtested Total Return (full period)",
            "Optimized Portfolio (Max Sharpe)": f"{opt_summary['total_return_pct']:.1f}%",
            "Benchmark": f"{bench_summary['total_return_pct']:.1f}%",
        },
        {
            "Metric": "Backtested Annualized Return",
            "Optimized Portfolio (Max Sharpe)": f"{opt_summary['annual_return_pct']:.2f}%",
            "Benchmark": f"{bench_summary['annual_return_pct']:.2f}%",
        },
        {
            "Metric": "Backtested Annualized Volatility",
            "Optimized Portfolio (Max Sharpe)": f"{opt_summary['annual_vol_pct']:.2f}%",
            "Benchmark": f"{bench_summary['annual_vol_pct']:.2f}%",
        },
        {
            "Metric": "1-Day Historical VaR (95%)",
            "Optimized Portfolio (Max Sharpe)": f"{opt_summary['daily_var_95_pct']:.2f}%",
            "Benchmark": f"{bench_summary['daily_var_95_pct']:.2f}%",
        },
        {
            "Metric": "1-Day Historical CVaR (95%)",
            "Optimized Portfolio (Max Sharpe)": f"{opt_summary['daily_cvar_95_pct']:.2f}%",
            "Benchmark": f"{bench_summary['daily_cvar_95_pct']:.2f}%",
        },
        {
            "Metric": "Annualized VaR (95%, sqrt-time scaled)",
            "Optimized Portfolio (Max Sharpe)": f"{opt_summary['annual_var_95_pct']:.2f}%",
            "Benchmark": f"{bench_summary['annual_var_95_pct']:.2f}%",
        },
        {
            "Metric": "Annualized CVaR (95%, sqrt-time scaled)",
            "Optimized Portfolio (Max Sharpe)": f"{opt_summary['annual_cvar_95_pct']:.2f}%",
            "Benchmark": f"{bench_summary['annual_cvar_95_pct']:.2f}%",
        },
        {
            "Metric": "Maximum Drawdown",
            "Optimized Portfolio (Max Sharpe)": f"{opt_summary['max_drawdown_pct']:.2f}%",
            "Benchmark": f"{bench_summary['max_drawdown_pct']:.2f}%",
        },
    ]
    pd.DataFrame(summary_rows).to_csv(os.path.join(RESULTS_DIR, "results_summary.csv"), index=False)


if __name__ == "__main__":
    main()
