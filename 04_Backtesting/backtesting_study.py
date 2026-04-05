"""
Backtesting Study — Risk-Based Optimal Portfolios
====================================================
Assignment:
  "Using the historical data, setup backtesting study for three risk-based
   optimal portfolios (equal weight, min vol, max diversification) for
   comparison and recommendation."

STRATEGIES
----------
  1. Equal Weight (EW):  w_i = 1/N  — naïve diversification baseline.
  2. Minimum Volatility (MinVol):  min wᵀΣw  s.t. Σw=1, w≥0.
     Finds the portfolio with the lowest possible risk using only the
     covariance structure.  Ignores expected returns entirely.
  3. Maximum Diversification (MaxDiv):  max (wᵀσ) / √(wᵀΣw)  s.t. Σw=1, w≥0.
     Maximizes the Diversification Ratio — the ratio of the weighted-average
     individual volatility to the portfolio volatility.  A DR > 1 means the
     portfolio benefits from imperfect correlations (diversification).
     (Choueifaty & Coignard, 2008)

BACKTESTING METHODOLOGY
-----------------------
  For each strategy we run TWO backtesting approaches:

  (a) Rolling Window (36-month lookback):
      At each rebalancing date t, estimate Σ from returns [t-36, t-1].
      Fixed window adapts faster to regime changes but is noisier.

  (b) Expanding Window:
      At each rebalancing date t, estimate Σ from returns [0, t-1].
      Uses all available history — more stable, slower to react.

  Both approaches:
    - Rebalance monthly (apply new weights to next month's realized return).
    - Start after a minimum burn-in period of 36 months.
    - Are fully out-of-sample: weights are computed BEFORE observing the
      return they will be applied to.

OUTPUT
------
  • Console:  Performance summary table, analysis, recommendation
  • CSV:     backtest_performance_summary.csv, backtest_cumulative_returns.csv,
             backtest_weights_history.csv
  • Charts:  backtest_cumulative_wealth.png, backtest_rolling_sharpe.png,
             backtest_weight_allocations.png, backtest_drawdowns.png
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend for saving charts
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ==============================================================================
# CONFIGURATION
# ==============================================================================
LOOKBACK = 36  # Rolling window size (months) — 3 years
MIN_HISTORY = 36  # Minimum months of data before first trade
REBAL_FREQ = 1  # Rebalance every N months (1 = monthly)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "Data", "monthly_returns.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "Outputs")

# Chart styling — premium academic look
plt.rcParams.update(
    {
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
    }
)

# Strategy color palette (vibrant, distinguishable)
COLORS = {
    "EW_rolling": "#4fc3f7",  # Light blue
    "EW_expanding": "#0288d1",  # Dark blue
    "MinVol_rolling": "#aed581",  # Light green
    "MinVol_expanding": "#558b2f",  # Dark green
    "MaxDiv_rolling": "#ffb74d",  # Light orange
    "MaxDiv_expanding": "#e65100",  # Dark orange
}

LABELS = {
    "EW_rolling": "Equal Weight  (Rolling 36m)",
    "EW_expanding": "Equal Weight  (Expanding)",
    "MinVol_rolling": "Min Volatility  (Rolling 36m)",
    "MinVol_expanding": "Min Volatility  (Expanding)",
    "MaxDiv_rolling": "Max Diversification  (Rolling 36m)",
    "MaxDiv_expanding": "Max Diversification  (Expanding)",
}

LINESTYLES = {
    "EW_rolling": "-",
    "EW_expanding": "--",
    "MinVol_rolling": "-",
    "MinVol_expanding": "--",
    "MaxDiv_rolling": "-",
    "MaxDiv_expanding": "--",
}


# ==============================================================================
# LOAD DATA
# ==============================================================================
df = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
ASSETS = df.columns.tolist()
N = len(ASSETS)

print("=" * 80)
print("BACKTESTING STUDY -- Risk-Based Optimal Portfolios")
print("=" * 80)
print(f"\nAssets: {ASSETS}")
print(f"Data:   {df.index.min().date()} to {df.index.max().date()}  ({len(df)} months)")
print(
    f"Config: lookback={LOOKBACK}m, min_history={MIN_HISTORY}m, rebal_freq={REBAL_FREQ}m"
)


# ==============================================================================
# STRATEGY FUNCTIONS
# ==============================================================================
# Each function takes a covariance matrix (annualized) and returns weight vector.


def equal_weight(cov_matrix):
    """Equal Weight: w_i = 1/N.

    The simplest possible allocation.  No estimation error because it uses
    NO parameters at all.  Surprisingly hard to beat in practice (DeMiguel
    et al., 2009 — "1/N" paper).
    """
    n = cov_matrix.shape[0]
    return np.ones(n) / n


def min_volatility(cov_matrix):
    """Minimum Volatility: min wᵀΣw  s.t. Σw=1, w≥0.

    WHY IT WORKS:
      The minimum-variance portfolio exploits low-volatility assets AND
      negative/low correlations to achieve the lowest possible risk.
      It is the leftmost point on the efficient frontier.

    WHY NO EXPECTED RETURNS?
      MinVol deliberately ignores return estimates because expected returns
      are notoriously hard to estimate accurately.  By focusing ONLY on risk
      (which is more stable and predictable), MinVol avoids the estimation
      error amplification that plagues traditional MVO.

    MATHEMATICAL FORMULATION:
      minimize    wᵀ Σ w                (portfolio variance)
      subject to  Σ wᵢ = 1              (fully invested)
                  wᵢ ≥ 0 ∀i            (no short-selling)
    """
    n = cov_matrix.shape[0]
    w0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    result = minimize(
        fun=lambda w: w @ cov_matrix @ w,  # portfolio variance (not vol)
        x0=w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    if result.success:
        # Clip numerical noise and renormalize
        w = np.maximum(result.x, 0.0)
        w /= w.sum()
        return w
    else:
        # Fallback to equal weight if solver fails
        return np.ones(n) / n


def max_diversification(cov_matrix):
    """Maximum Diversification: max DR = (wᵀσ) / √(wᵀΣw)  s.t. Σw=1, w≥0.

    DIVERSIFICATION RATIO (DR):
      DR = (Σ wᵢ σᵢ) / σ_p
         = (weighted average of individual vols) / (portfolio vol)

    INTERPRETATION:
      • DR = 1  →  no diversification benefit (e.g., all assets perfectly correlated)
      • DR > 1  →  portfolio vol is LESS than the weighted-avg of individual vols
                    because of imperfect correlations → diversification is working
      • Higher DR →  more diversification → risk is more evenly spread across
                     uncorrelated sources

    WHY MAXDIV?
      Unlike MinVol which concentrates in the lowest-vol assets, MaxDiv
      rewards allocating to ANY asset that contributes diversification,
      even high-vol ones — as long as they are uncorrelated with the rest.
      This typically produces a more balanced portfolio.

    MATHEMATICAL FORMULATION:
      maximize    (wᵀ σ) / √(wᵀ Σ w)   (Diversification Ratio)
      subject to  Σ wᵢ = 1              (fully invested)
                  wᵢ ≥ 0 ∀i            (no short-selling)

    NOTE: We minimize the NEGATIVE DR (scipy only minimizes).
    """
    n = cov_matrix.shape[0]
    sigma = np.sqrt(np.diag(cov_matrix))  # individual asset volatilities
    w0 = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def neg_diversification_ratio(w):
        # Denominator (Slide 1): Total portfolio volatility taking into account correlations
        port_vol = np.sqrt(w @ cov_matrix @ w)
        if port_vol < 1e-12:
            return 0.0

        # Numerator (Slide 1): Weighted sum of underlying asset volatilities
        weighted_avg_vol = w @ sigma

        # We want to MAXIMIZE this ratio, so we MINIMIZE its negative for SciPy
        # Both this formulation and the QP correlation pseudo-inverse (Slide 2) converge to the exact same weights.
        return -(weighted_avg_vol / port_vol)

    result = minimize(
        fun=neg_diversification_ratio,
        x0=w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )

    if result.success:
        w = np.maximum(result.x, 0.0)
        w /= w.sum()
        return w
    else:
        return np.ones(n) / n


# ==============================================================================
# BACKTESTING ENGINE
# ==============================================================================
def run_backtest(returns_df, strategy_fn, window_type, lookback=LOOKBACK):
    """Run a single backtest for one strategy × one window type.

    Parameters
    ----------
    returns_df : pd.DataFrame
        Monthly returns, shape (T, N).  Each row is one month.
    strategy_fn : callable
        Function(cov_matrix) → weight vector of length N.
    window_type : str
        "rolling" or "expanding".
    lookback : int
        Number of months for the rolling window (ignored for expanding).

    Returns
    -------
    portfolio_returns : pd.Series
        Monthly portfolio returns (out-of-sample).
    weights_history : pd.DataFrame
        Weight allocations at each rebalancing date, shape (T_oos, N).
    turnover_series : pd.Series
        Monthly turnover (sum of absolute weight changes).

    BACKTEST LOGIC (step by step):
    ──────────────────────────────────────────────────────────────────
    For each month t (starting after burn-in):
      1. ESTIMATION: Compute Σ from returns in the estimation window:
         - Rolling:   returns[t - lookback : t]
         - Expanding: returns[0 : t]
      2. OPTIMIZATION: Apply strategy_fn(Σ) → target weights w_target
      3. APPLICATION: Portfolio return at t = w_target ᵀ × r_t
         (weights determined BEFORE observing r_t → no look-ahead bias)
      4. DRIFT: After r_t is realized, weights drift due to returns.
         New weights = w_target * (1 + r_t) / Σ(w_target * (1 + r_t))
         Turnover = Σ |w_target_new - w_drifted|
    """
    T = len(returns_df)
    n = returns_df.shape[1]
    dates = returns_df.index

    # Minimum start index: need at least MIN_HISTORY months of data
    start_idx = max(MIN_HISTORY, lookback) if window_type == "rolling" else MIN_HISTORY

    port_returns = []
    port_dates = []
    weight_rows = []
    turnover_vals = []
    prev_weights = None

    for t in range(start_idx, T):
        # Step 1: Define estimation window
        if window_type == "rolling":
            est_start = t - lookback
            est_data = returns_df.iloc[est_start:t]
        else:  # expanding
            est_data = returns_df.iloc[:t]

        # Step 2: Estimate covariance matrix (annualized)
        cov_est = est_data.cov().values * 12

        # Regularization: add small ridge to diagonal for numerical stability
        # WHY: With only 36 months and 4 assets, the sample covariance can be
        # near-singular in degenerate cases.  Adding ε to the diagonal prevents
        # optimizer failures without materially affecting results.
        cov_est += np.eye(n) * 1e-8

        # Step 3: Compute optimal weights
        w = strategy_fn(cov_est)

        # Step 4: Apply weights to next month's realized return
        r_t = returns_df.iloc[t].values  # realized return vector at time t
        port_ret = w @ r_t  # portfolio return

        port_returns.append(port_ret)
        port_dates.append(dates[t])
        weight_rows.append(w.copy())

        # Step 5: Compute turnover (if not first period)
        if prev_weights is not None:
            # Drift previous weights by last period's return
            w_drifted = prev_weights * (1 + r_t)
            w_drifted /= w_drifted.sum()
            turnover = np.sum(np.abs(w - w_drifted))
            turnover_vals.append(turnover)
        else:
            turnover_vals.append(0.0)

        prev_weights = w.copy()

    portfolio_returns = pd.Series(port_returns, index=port_dates, name="PortReturn")
    weights_history = pd.DataFrame(
        weight_rows, index=port_dates, columns=returns_df.columns
    )
    turnover_series = pd.Series(turnover_vals, index=port_dates, name="Turnover")

    return portfolio_returns, weights_history, turnover_series


# ==============================================================================
# PERFORMANCE METRICS
# ==============================================================================
def compute_metrics(returns_series, turnover_series=None):
    """Compute comprehensive performance metrics for a return series.

    All metrics are annualized (monthly returns × 12 or × √12 as appropriate).

    Returns
    -------
    dict with keys:
      Ann_Return    : annualized geometric mean return
      Ann_Vol       : annualized volatility (std dev)
      Sharpe        : Sharpe ratio (Rf = 0)
      Max_Drawdown  : maximum peak-to-trough decline
      Calmar        : Ann_Return / |Max_Drawdown|
      Avg_Turnover  : average monthly one-way turnover
      Final_Wealth  : terminal value of $1 invested
    """
    # Geometric annualized return
    cumulative = (1 + returns_series).prod()
    n_years = len(returns_series) / 12
    ann_ret = cumulative ** (1 / n_years) - 1 if n_years > 0 else 0

    # Annualized volatility
    ann_vol = returns_series.std() * np.sqrt(12)

    # Sharpe ratio (Rf = 0)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    # Maximum drawdown
    cum_wealth = (1 + returns_series).cumprod()
    running_max = cum_wealth.cummax()
    drawdown = (cum_wealth - running_max) / running_max
    max_dd = drawdown.min()

    # Calmar ratio
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.inf

    # Average turnover
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
# RUN ALL BACKTESTS
# ==============================================================================
STRATEGIES = {
    "EW": equal_weight,
    "MinVol": min_volatility,
    "MaxDiv": max_diversification,
}

WINDOW_TYPES = ["rolling", "expanding"]

print(f"\n{'─' * 80}")
print("RUNNING BACKTESTS...")
print(f"{'─' * 80}")

all_returns = {}
all_weights = {}
all_turnover = {}
all_metrics = {}

for strat_name, strat_fn in STRATEGIES.items():
    for wtype in WINDOW_TYPES:
        key = f"{strat_name}_{wtype}"
        print(f"  {LABELS[key]:45s} ... ", end="", flush=True)

        port_ret, wgt_hist, turnover = run_backtest(df, strat_fn, wtype)

        all_returns[key] = port_ret
        all_weights[key] = wgt_hist
        all_turnover[key] = turnover

        metrics = compute_metrics(port_ret, turnover)
        all_metrics[key] = metrics

        print(
            f"Sharpe={metrics['Sharpe']:+.3f}  "
            f"Ret={metrics['Ann_Return']*100:+5.1f}%  "
            f"Vol={metrics['Ann_Vol']*100:5.1f}%  "
            f"MaxDD={metrics['Max_Drawdown']*100:5.1f}%"
        )


# ==============================================================================
# PERFORMANCE SUMMARY TABLE
# ==============================================================================
print(f"\n\n{'=' * 80}")
print("PERFORMANCE SUMMARY -- All Strategies x Window Types")
print(f"{'=' * 80}")

# Determine OOS period from the first strategy
first_key = list(all_returns.keys())[0]
oos_start = all_returns[first_key].index[0].strftime("%Y-%m")
oos_end = all_returns[first_key].index[-1].strftime("%Y-%m")
oos_months = len(all_returns[first_key])
print(f"Out-of-sample period: {oos_start} to {oos_end}  ({oos_months} months)\n")

summary_rows = []
for key, metrics in all_metrics.items():
    row = {"Strategy": LABELS[key]}
    row.update(
        {
            "Ann. Return": f"{metrics['Ann_Return']*100:+.2f}%",
            "Ann. Vol": f"{metrics['Ann_Vol']*100:.2f}%",
            "Sharpe": f"{metrics['Sharpe']:.3f}",
            "Max DD": f"{metrics['Max_Drawdown']*100:.1f}%",
            "Calmar": f"{metrics['Calmar']:.3f}",
            "Avg Turnover": f"{metrics['Avg_Turnover']*100:.2f}%",
            "Final $1": f"${metrics['Final_Wealth']:.2f}",
        }
    )
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows).set_index("Strategy")
print(summary_df.to_string())

# Save summary
summary_csv_path = os.path.join(OUT_DIR, "backtest_performance_summary.csv")
# Also save numeric version
numeric_rows = []
for key, metrics in all_metrics.items():
    row = {"Strategy": LABELS[key]}
    row.update(metrics)
    numeric_rows.append(row)
pd.DataFrame(numeric_rows).set_index("Strategy").to_csv(summary_csv_path)


# ==============================================================================
# ROLLING vs EXPANDING WINDOW COMPARISON
# ==============================================================================
print(f"\n\n{'=' * 80}")
print("ROLLING vs EXPANDING WINDOW COMPARISON")
print(f"{'=' * 80}")
print(
    """
  Rolling Window (36-month lookback):
    [+] Adapts faster to regime changes (GFC, COVID)
    [+] Weights reflect CURRENT market conditions
    [-] Higher estimation noise (smaller sample)
    [-] Higher turnover (weights change more frequently)

  Expanding Window:
    [+] More stable estimates (larger sample over time)
    [+] Lower turnover
    [-] Slower to react to regime shifts
    [-] Early periods dominated by small samples
"""
)

for strat in STRATEGIES:
    r_key = f"{strat}_rolling"
    e_key = f"{strat}_expanding"
    rm = all_metrics[r_key]
    em = all_metrics[e_key]
    print(
        f"  {strat:8s}  Rolling SR={rm['Sharpe']:.3f}  Expanding SR={em['Sharpe']:.3f}  "
        f"-> {'Rolling' if rm['Sharpe'] > em['Sharpe'] else 'Expanding'} wins"
    )


# ==============================================================================
# CROSS-STRATEGY ANALYSIS
# ==============================================================================
print(f"\n\n{'=' * 80}")
print("CROSS-STRATEGY ANALYSIS (Rolling Window)")
print(f"{'=' * 80}")

print(
    """
  Equal Weight (EW):
    • Zero estimation error — no optimization required.
    • Implicitly assumes all assets have equal risk & correlation.
    • Strong baseline; often wins due to diversification across
      all assets without concentrating in any one.

  Minimum Volatility (MinVol):
    • Uses ONLY the covariance matrix (ignores returns).
    • Concentrates in low-vol assets (typically bonds).
    • Best for risk-averse investors or capital preservation.
    • Risk: concentration in bonds can hurt during rate-hike regimes.

  Maximum Diversification (MaxDiv):
    • Maximizes the Diversification Ratio.
    • Rewards spreading risk across uncorrelated assets.
    • More balanced than MinVol; allocates to high-vol assets IF
      they contribute diversification.
    • Best risk-adjusted returns in many empirical studies.
"""
)


# ==============================================================================
# SAVE CUMULATIVE RETURNS CSV
# ==============================================================================
cum_returns_df = pd.DataFrame(
    {LABELS[k]: (1 + v).cumprod() for k, v in all_returns.items()}
)
cum_returns_path = os.path.join(OUT_DIR, "backtest_cumulative_returns.csv")
cum_returns_df.to_csv(cum_returns_path)

# Save weights history
weights_dfs = []
for key, wgt in all_weights.items():
    wgt_copy = wgt.copy()
    wgt_copy.insert(0, "Strategy", LABELS[key])
    weights_dfs.append(wgt_copy)
weights_all = pd.concat(weights_dfs)
weights_path = os.path.join(OUT_DIR, "backtest_weights_history.csv")
weights_all.to_csv(weights_path)


# ==============================================================================
# CHART 1: CUMULATIVE WEALTH
# ==============================================================================
fig, ax = plt.subplots(figsize=(14, 7))

for key in all_returns:
    cum = (1 + all_returns[key]).cumprod()
    ax.plot(
        cum.index,
        cum.values,
        color=COLORS[key],
        linestyle=LINESTYLES[key],
        linewidth=1.8 if "rolling" in key else 1.2,
        alpha=0.9,
        label=LABELS[key],
    )

ax.set_title("Cumulative Wealth - $1 Invested", fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("Portfolio Value ($)")
ax.set_xlabel("")
ax.legend(loc="upper left", fontsize=8, framealpha=0.8)
ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.1f"))
ax.set_xlim(cum.index[0], cum.index[-1])

# Add recession shading for key events
crisis_periods = [
    ("2007-12-01", "2009-06-30", "GFC"),
    ("2020-02-01", "2020-04-30", "COVID"),
]
for start, end, label in crisis_periods:
    try:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.08, color="#ff5252")
    except Exception:
        pass

fig.tight_layout()
chart1_path = os.path.join(OUT_DIR, "backtest_cumulative_wealth.png")
fig.savefig(chart1_path)
plt.close(fig)
print(f"\n  Saved: {chart1_path}")


# ==============================================================================
# CHART 2: ROLLING 12-MONTH SHARPE RATIO
# ==============================================================================
fig, ax = plt.subplots(figsize=(14, 6))

ROLLING_SR_WINDOW = 12  # 12-month rolling Sharpe

for key in all_returns:
    ret = all_returns[key]
    rolling_mean = ret.rolling(ROLLING_SR_WINDOW).mean() * 12
    rolling_std = ret.rolling(ROLLING_SR_WINDOW).std() * np.sqrt(12)
    rolling_sr = rolling_mean / rolling_std
    rolling_sr = rolling_sr.dropna()

    ax.plot(
        rolling_sr.index,
        rolling_sr.values,
        color=COLORS[key],
        linestyle=LINESTYLES[key],
        linewidth=1.4,
        alpha=0.85,
        label=LABELS[key],
    )

ax.axhline(y=0, color="#555555", linewidth=0.8, linestyle=":")
ax.set_title("Rolling 12-Month Sharpe Ratio", fontsize=14, fontweight="bold", pad=12)
ax.set_ylabel("Sharpe Ratio")
ax.set_xlabel("")
ax.legend(loc="lower left", fontsize=8, framealpha=0.8)
ax.set_ylim(-4, 6)

for start, end, label in crisis_periods:
    try:
        ax.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.08, color="#ff5252")
    except Exception:
        pass

fig.tight_layout()
chart2_path = os.path.join(OUT_DIR, "backtest_rolling_sharpe.png")
fig.savefig(chart2_path)
plt.close(fig)
print(f"  Saved: {chart2_path}")


# ==============================================================================
# CHART 3: WEIGHT ALLOCATIONS (Rolling window strategies only — stacked area)
# ==============================================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

# Color palette for assets
asset_colors = ["#4fc3f7", "#66bb6a", "#ffb74d", "#ef5350"]

strat_order = ["EW", "MinVol", "MaxDiv"]
strat_titles = [
    "Equal Weight (Rolling 36m)",
    "Min Volatility (Rolling 36m)",
    "Max Diversification (Rolling 36m)",
]

for idx, (strat, title) in enumerate(zip(strat_order, strat_titles)):
    ax = axes[idx]
    key = f"{strat}_rolling"
    wgt = all_weights[key]

    ax.stackplot(
        wgt.index,
        [wgt[a].values for a in ASSETS],
        labels=ASSETS,
        colors=asset_colors,
        alpha=0.8,
    )
    ax.set_ylabel("Weight")
    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))

    if idx == 0:
        ax.legend(loc="upper right", fontsize=8, ncol=4, framealpha=0.8)

fig.suptitle(
    "Portfolio Weight Allocations Over Time", fontsize=14, fontweight="bold", y=0.98
)
fig.tight_layout()
chart3_path = os.path.join(OUT_DIR, "backtest_weight_allocations.png")
fig.savefig(chart3_path)
plt.close(fig)
print(f"  Saved: {chart3_path}")


# ==============================================================================
# CHART 4: DRAWDOWNS (separate panels per strategy for clarity)
# ==============================================================================
fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

strategy_groups = [
    ("Equal Weight", ["EW_rolling", "EW_expanding"], "#4fc3f7"),
    ("Min Volatility", ["MinVol_rolling", "MinVol_expanding"], "#66bb6a"),
    ("Max Diversification", ["MaxDiv_rolling", "MaxDiv_expanding"], "#ffb74d"),
]

for ax_idx, (strat_label, keys, base_color) in enumerate(strategy_groups):
    ax = axes[ax_idx]

    for key in keys:
        cum = (1 + all_returns[key]).cumprod()
        running_max = cum.cummax()
        dd = (cum - running_max) / running_max

        is_rolling = "rolling" in key
        alpha_fill = 0.3 if is_rolling else 0.15
        lw = 2.0 if is_rolling else 1.5
        ls = "-" if is_rolling else "--"

        ax.fill_between(dd.index, dd.values, 0, color=base_color, alpha=alpha_fill)
        ax.plot(
            dd.index,
            dd.values,
            color=COLORS[key],
            linestyle=ls,
            linewidth=lw,
            label=LABELS[key],
        )

        # Annotate worst drawdown
        worst_idx = dd.idxmin()
        worst_val = dd.min()
        if is_rolling:
            ax.annotate(
                f"{worst_val*100:.1f}%",
                xy=(worst_idx, worst_val),
                xytext=(10, -15),
                textcoords="offset points",
                fontsize=10,
                fontweight="bold",
                color=COLORS[key],
                arrowprops=dict(arrowstyle="->", color=COLORS[key], lw=1.5),
            )

    ax.set_title(strat_label, fontsize=13, fontweight="bold", pad=8)
    ax.set_ylabel("Drawdown")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.legend(loc="lower left", fontsize=9, framealpha=0.8)
    ax.set_ylim(None, 0.02)
    ax.grid(True, linestyle=":", alpha=0.4)

    for start, end, label in crisis_periods:
        try:
            ax.axvspan(
                pd.Timestamp(start), pd.Timestamp(end), alpha=0.10, color="#ff5252"
            )
            mid = pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2
            if ax_idx == 0:
                ax.text(
                    mid,
                    0.015,
                    label,
                    ha="center",
                    fontsize=8,
                    color="#ff5252",
                    fontweight="bold",
                    alpha=0.8,
                )
        except Exception:
            pass

fig.suptitle(
    "Drawdown Analysis — Strategy Comparison", fontsize=15, fontweight="bold", y=1.01
)
fig.tight_layout()
chart4_path = os.path.join(OUT_DIR, "backtest_drawdowns.png")
fig.savefig(chart4_path, bbox_inches="tight")
plt.close(fig)
print(f"  Saved: {chart4_path}")


# ==============================================================================
# RECOMMENDATION & CONCLUSIONS
# ==============================================================================
print(f"\n\n{'=' * 80}")
print("CONCLUSIONS & RECOMMENDATION")
print("=" * 80)

# Find best strategy by Sharpe
best_key = max(all_metrics, key=lambda k: all_metrics[k]["Sharpe"])
best_label = LABELS[best_key]
best_m = all_metrics[best_key]

# Find most defensive (lowest max drawdown)
safest_key = max(
    all_metrics, key=lambda k: all_metrics[k]["Max_Drawdown"]
)  # least negative
safest_label = LABELS[safest_key]
safest_m = all_metrics[safest_key]

print(
    """
  -- KEY FINDINGS ----------------------------------------------------------

  1. BEST RISK-ADJUSTED RETURN (Sharpe):"""
)
print(f"     {best_label}")
print(
    f"     Sharpe = {best_m['Sharpe']:.3f},  Ann. Return = {best_m['Ann_Return']*100:+.2f}%,"
)
print(
    f"     Ann. Vol = {best_m['Ann_Vol']*100:.2f}%,  Max DD = {best_m['Max_Drawdown']*100:.1f}%"
)
print(
    """

  2. MOST DEFENSIVE (Smallest Drawdown):"""
)
print(f"     {safest_label}")
print(
    f"     Max DD = {safest_m['Max_Drawdown']*100:.1f}%,  Sharpe = {safest_m['Sharpe']:.3f}"
)
print(
    """

  3. ROLLING vs EXPANDING:
     Rolling window adapts to regime changes (better during transitions).
     Expanding window provides more stable weights (lower turnover).
     Neither dominates universally — choice depends on investor belief
     about regime persistence.

  -- STRATEGY COMPARISON ---------------------------------------------------

  Equal Weight (EW):
    • Simplest — no optimization needed, no estimation error.
    • Strong diversification baseline, often competitive.
    • Best when all assets have similar risk-adjusted returns.

  Minimum Volatility (MinVol):
    • Lowest-risk portfolio; best for capital preservation.
    • Heavy tilt toward bonds/low-vol assets.
    • Underperforms in bull markets; excels in crises.
    • Risk: concentration can hurt during rate hikes.

  Maximum Diversification (MaxDiv):
    • Spreads risk most evenly across uncorrelated sources.
    • More balanced than MinVol; includes high-vol assets if they
      contribute diversification.
    • Best theoretical justification for a risk-based approach.

  -- RECOMMENDATION --------------------------------------------------------

  For a LONG-TERM INVESTOR seeking robust risk-adjusted returns:
    -> Maximum Diversification (Rolling) is the recommended strategy.
       It combines diversification benefits with regime adaptation.

  For a RISK-AVERSE / CAPITAL PRESERVATION mandate:
    -> Minimum Volatility (Expanding) provides the lowest drawdowns
      with stable, predictable allocations.

  For SIMPLICITY and TRANSPARENCY:
    -> Equal Weight remains a surprisingly strong benchmark.
       Use it as your "default" and measure other strategies against it.
"""
)

# Save summary files list
print(f"\n{'─' * 80}")
print("OUTPUT FILES:")
print(f"{'─' * 80}")
print(f"  CSV:    {summary_csv_path}")
print(f"  CSV:    {cum_returns_path}")
print(f"  CSV:    {weights_path}")
print(f"  Chart:  {chart1_path}")
print(f"  Chart:  {chart2_path}")
print(f"  Chart:  {chart3_path}")
print(f"  Chart:  {chart4_path}")
print()
