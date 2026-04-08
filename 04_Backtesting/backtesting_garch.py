"""
GARCH-Based Backtesting Study — Risk-Based Portfolios
=====================================================
Upgrades the traditional backtesting methodology by using a GARCH(1,1) 
model to forecast next-month volatility, capturing "volatility clustering", 
combined with a sample correlation matrix to build the forecasted covariance.

Optimization Targets:
1. Equal Weight (EW) — purely naive
2. Minimum Volatility (MinVol) — risk minimization
3. Maximum Diversification (MaxDiv) — best risk spread

Outputs charts and metrics for the Investor Proposal.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from arch import arch_model
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import warnings

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIGURATION
# ==============================================================================
# Using 60-month lookback here to ensure enough data points for GARCH convergence 
# (GARCH generally needs more data than simple sample covariance).
LOOKBACK = 60
MIN_HISTORY = 60
REBAL_FREQ = 1

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "Data", "monthly_returns.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "Outputs")

# Styling for Charts
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "#cccccc",
    "axes.labelcolor": "#333333",
    "axes.grid": True,
    "grid.color": "#e0e0e0",
    "grid.alpha": 0.6,
    "text.color": "#333333",
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "legend.facecolor": "white",
    "legend.edgecolor": "#cccccc",
    "legend.labelcolor": "#333333",
    "font.family": "sans-serif",
    "font.size": 10,
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.3,
})

COLORS = {
    "EW_rolling":       "#4fc3f7",
    "EW_expanding":     "#0288d1",
    "MinVol_rolling":   "#aed581",
    "MinVol_expanding": "#558b2f",
    "MaxDiv_rolling":   "#ffb74d",
    "MaxDiv_expanding": "#e65100",
}

LABELS = {
    "EW_rolling":       "Equal Weight (Rolling 60m)",
    "EW_expanding":     "Equal Weight (Expanding)",
    "MinVol_rolling":   "Min Volatility (Rolling 60m)",
    "MinVol_expanding": "Min Volatility (Expanding)",
    "MaxDiv_rolling":   "Max Div. (Rolling 60m)",
    "MaxDiv_expanding": "Max Div. (Expanding)",
}

LINESTYLES = {
    "EW_rolling":       "-",
    "EW_expanding":     "--",
    "MinVol_rolling":   "-",
    "MinVol_expanding": "--",
    "MaxDiv_rolling":   "-",
    "MaxDiv_expanding": "--",
}

# ==============================================================================
# STRATEGY FUNCTIONS
# ==============================================================================
def equal_weight(cov_matrix):
    n = cov_matrix.shape[0]
    return np.ones(n) / n

def min_volatility(cov_matrix):
    n = cov_matrix.shape[0]
    w0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    res = minimize(
        fun=lambda w: w @ cov_matrix @ w,
        x0=w0, method="SLSQP", bounds=bounds, constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000}
    )
    if res.success:
        w = np.maximum(res.x, 0.0)
        return w / w.sum()
    return w0

def max_diversification(cov_matrix):
    """
    Maximum Diversification Portfolio (MDP)
    Objective: Maximize the Diversification Ratio = (w.T * sigma) / sqrt(w.T * Cov * w)
    
    Note: Both this direct non-linear formulation (from Slide 1) and the QP formulation 
    using the correlation matrix (from Slide 2) mathematically converge to the 
    exact same optimal weights. We use the direct ratio formulation because it is 
    computationally trivial for N=4 and highly readable.
    """
    n = cov_matrix.shape[0]
    sigma = np.sqrt(np.diag(cov_matrix))  # Extract individual standard deviations (sigma_i)
    
    w0 = np.ones(n) / n  # Initial guess (equal weights)
    bounds = [(0.0, 1.0)] * n  # No short selling (Long only)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]  # Fully invested constraint

    def neg_div_ratio(w):
        # Denominator: Total portfolio volatility taking into account correlations
        port_vol = np.sqrt(w @ cov_matrix @ w)
        if port_vol < 1e-12: return 0.0
        
        # Numerator: Weighted sum of underlying asset volatilities
        weighted_avg_vol = w @ sigma
        
        # We want to MAXIMIZE this ratio, so we MINIMIZE its negative for SciPy
        return -(weighted_avg_vol / port_vol)

    res = minimize(
        fun=neg_div_ratio, x0=w0, method="SLSQP", bounds=bounds,
        constraints=constraints, options={"ftol": 1e-12, "maxiter": 1000}
    )
    if res.success:
        w = np.maximum(res.x, 0.0)
        return w / w.sum()
    return w0

# ==============================================================================
# GARCH COVARIANCE ESTIMATION
# ==============================================================================
def forecast_covariance(est_data):
    """
    Fits a GARCH(1,1) model for each asset to forecast next month's volatility.
    Combines these with the sample correlation matrix to produce a forecasted
    covariance matrix. Returns annualized covariance matrix.
    """
    n = est_data.shape[1]
    
    # 1. Calculate sample correlation matrix
    sample_corr = est_data.corr().values
    
    # 2. Forecast next month's variance for each asset using GARCH(1,1)
    garch_vols = []
    
    for i in range(n):
        returns = est_data.iloc[:, i] * 100  # arch is numerically unstable with small returns
        # rescale to percentages for fitting
        
        # Fit a Standard GARCH(1,1) with a constant mean
        try:
            am = arch_model(returns, vol='Garch', p=1, q=1, mean='Constant')
            res = am.fit(disp="off", show_warning=False)
            
            # Forecast variance for step t+1 (divide by 10000 to reverse % scaling)
            forecasts = res.forecast(horizon=1, align='origin')
            next_var = forecasts.variance.iloc[-1, 0] / 10000.0
            
            # Annualize and take square root for volatility
            ann_vol = np.sqrt(next_var * 12)
            garch_vols.append(ann_vol)
        except Exception:
            # Fallback to sample volatility if GARCH fails to converge
            ann_vol = returns.std() * np.sqrt(12) / 100.0
            garch_vols.append(ann_vol)
            
    garch_vols = np.array(garch_vols)
    D = np.diag(garch_vols)
    
    # Sigma = D * R * D
    forecasted_cov = D @ sample_corr @ D
    
    # Add regularizer
    forecasted_cov += np.eye(n) * 1e-8
    return forecasted_cov

# ==============================================================================
# BACKTESTING ENGINE
# ==============================================================================
def run_backtest_garch(returns_df, strategy_fn, window_type, lookback=LOOKBACK):
    T = len(returns_df)
    n = returns_df.shape[1]
    dates = returns_df.index
    
    start_idx = max(MIN_HISTORY, lookback) if window_type == "rolling" else MIN_HISTORY
    
    port_returns = []
    port_dates = []
    weight_rows = []
    turnover_vals = []
    prev_weights = None

    for t in range(start_idx, T):
        if window_type == "rolling":
            est_data = returns_df.iloc[t - lookback : t]
        else:
            est_data = returns_df.iloc[:t]
            
        # Instead of simple sample covariance, use GARCH-based structural covariance
        if strategy_fn == equal_weight:
            # Short-circuit for EW, cov isn't needed
            cov_est = np.eye(n) 
        else:
            cov_est = forecast_covariance(est_data)
        
        w = strategy_fn(cov_est)
        r_t = returns_df.iloc[t].values
        port_ret = w @ r_t
        
        port_returns.append(port_ret)
        port_dates.append(dates[t])
        weight_rows.append(w.copy())
        
        if prev_weights is not None:
            w_opt = prev_weights * (1 + r_t)
            w_drifted = w_opt / (w_opt.sum() if w_opt.sum() != 0 else 1e-12)
            turnover = np.sum(np.abs(w - w_drifted))
            turnover_vals.append(turnover)
        else:
            turnover_vals.append(0.0)
            
        prev_weights = w.copy()
        
    return (pd.Series(port_returns, index=port_dates, name="PortReturn"),
            pd.DataFrame(weight_rows, index=port_dates, columns=returns_df.columns),
            pd.Series(turnover_vals, index=port_dates, name="Turnover"))

# ==============================================================================
# PERFORMANCE METRICS
# ==============================================================================
def compute_metrics(returns_series, turnover_series=None):
    cumulative = (1 + returns_series).prod()
    n_years = len(returns_series) / 12
    ann_ret = cumulative ** (1 / n_years) - 1 if n_years > 0 else 0
    ann_vol = returns_series.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    
    cum_wealth = (1 + returns_series).cumprod()
    running_max = cum_wealth.cummax()
    max_dd = ((cum_wealth - running_max) / running_max).min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.inf
    avg_turnover = turnover_series.mean() if turnover_series is not None else 0
    
    return {
        "Ann_Return": ann_ret,
        "Ann_Vol": ann_vol,
        "Sharpe": sharpe,
        "Max_Drawdown": max_dd,
        "Calmar": calmar,
        "Avg_Turnover": avg_turnover,
        "Final_Wealth": cumulative,
    }

# ==============================================================================
# MAIN SCRIPT
# ==============================================================================
if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
    ASSETS = df.columns.tolist()
    
    STRATEGIES = {
        "EW": equal_weight,
        "MinVol": min_volatility,
        "MaxDiv": max_diversification,
    }
    WINDOW_TYPES = ["rolling", "expanding"]
    
    print("=" * 80)
    print("GARCH-BASED BACKTESTING STUDY (Robust Volatility Modeling)")
    print("=" * 80)
    print("Calculating GARCH-forecasted allocations. This may take a minute...")
    
    all_returns = {}
    all_weights = {}
    all_metrics = {}
    
    for strat_name, strat_fn in STRATEGIES.items():
        for wtype in WINDOW_TYPES:
            key = f"{strat_name}_{wtype}"
            print(f"Running: {LABELS[key]:40s}", flush=True)
            port_ret, wgt_hist, turnover = run_backtest_garch(df, strat_fn, wtype)
            all_returns[key] = port_ret
            all_weights[key] = wgt_hist
            metrics = compute_metrics(port_ret, turnover)
            all_metrics[key] = metrics
            
    print("\n" + "=" * 80)
    print("GARCH PERFORMANCE SUMMARY (OOS: 2005-2026)")
    print("=" * 80)
    
    summary_rows = []
    for key, m in all_metrics.items():
        row = {"Strategy": LABELS[key]}
        row.update({
            "Ann. Return": f"{m['Ann_Return']*100:+.2f}%",
            "Ann. Vol": f"{m['Ann_Vol']*100:.2f}%",
            "Sharpe": f"{m['Sharpe']:.3f}",
            "Max DD": f"{m['Max_Drawdown']*100:.1f}%",
            "Calmar": f"{m['Calmar']:.3f}",
            "Final $1": f"${m['Final_Wealth']:.2f}",
        })
        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows).set_index("Strategy")
    print(summary_df.to_string())
    
    # Saving Table
    summary_csv_path = os.path.join(OUT_DIR, "garch_backtest_summary.csv")
    pd.DataFrame([{**{"Strategy": LABELS[k]}, **v} for k, v in all_metrics.items()]).set_index("Strategy").to_csv(summary_csv_path)

    # 1. Cumulative Wealth Chart
    fig, ax = plt.subplots(figsize=(14, 7))
    for key in all_returns:
        cum = (1 + all_returns[key]).cumprod()
        ax.plot(cum.index, cum.values, color=COLORS[key], linestyle=LINESTYLES[key],
                linewidth=1.8 if "rolling" in key else 1.2, alpha=0.9, label=LABELS[key])
    
    ax.set_title("Capital Allocation Wealth - GARCH Forecasted ($1 Invested)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Portfolio Value ($)")
    ax.legend(loc="upper left", fontsize=8)
    
    crises = [("2007-12-01", "2009-06-30", "GFC"), ("2020-02-01", "2020-04-30", "COVID")]
    for start, end, label in crises:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.08, color="#ff5252")
        
    fig.savefig(os.path.join(OUT_DIR, "garch_cumulative_wealth.png"))
    plt.close(fig)

    # 2. Drawdowns Chart
    fig, ax = plt.subplots(figsize=(14, 6))
    for key in all_returns:
        cum = (1 + all_returns[key]).cumprod()
        dd = (cum - cum.cummax()) / cum.cummax()
        ax.plot(dd.index, dd.values, color=COLORS[key], linestyle=LINESTYLES[key],
                linewidth=1.2, alpha=0.9, label=LABELS[key])
        
    ax.set_title("Drawdown Analysis - Resilience Across Shocks", fontsize=14, fontweight="bold")
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="lower left", fontsize=8)
    for start, end, label in crises:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.08, color="#ff5252")
    
    fig.savefig(os.path.join(OUT_DIR, "garch_drawdowns.png"))
    plt.close(fig)
    
    print("\nSaved charts to: garch_cumulative_wealth.png, garch_drawdowns.png")
