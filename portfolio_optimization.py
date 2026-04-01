"""
Portfolio Optimization using Mean-Variance Optimization (MVO).

Uses full historical return/covariance assumptions with three objectives:
  1. Maximize return  s.t. annualized volatility ≤ 10%
  2. Minimize volatility  s.t. annualized return ≥ 4%
  3. Maximize Sharpe Ratio

All optimizations enforce: no short-selling (w_i ≥ 0), fully invested (Σw_i = 1).
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os

# ── Load Data ─────────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "monthly_returns.csv")
df = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
ASSETS = df.columns.tolist()
N = len(ASSETS)

# Annualized inputs
mu = df.mean().values * 12  # Expected returns (annualized)
cov = df.cov().values * 12  # Covariance matrix (annualized)

print("=" * 70)
print("PORTFOLIO OPTIMIZATION — Mean-Variance (Full Historical Assumptions)")
print("=" * 70)
print(f"\nAssets: {ASSETS}")
print(
    f"Data:   {df.index.min().date()} to {df.index.max().date()} ({len(df)} months)\n"
)

print("── Annualized Inputs ──")
for i, a in enumerate(ASSETS):
    print(
        f"  {a:25s}  μ = {mu[i]:+.4f} ({mu[i]*100:+.2f}%)  σ = {np.sqrt(cov[i,i]):.4f} ({np.sqrt(cov[i,i])*100:.2f}%)"
    )


# ── Helper Functions ──────────────────────────────────────────────────────────
def portfolio_return(w):
    return w @ mu


def portfolio_volatility(w):
    return np.sqrt(w @ cov @ w)


def portfolio_sharpe(w, rf=0.0):
    return (portfolio_return(w) - rf) / portfolio_volatility(w)


# Constraints & bounds (no short-selling)
bounds = [(0.0, 1.0)] * N
weight_sum_constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
w0 = np.ones(N) / N  # Equal-weight starting point


# ── Optimization 1: Max Return s.t. Vol ≤ 10% ────────────────────────────────
def opt1_max_return_vol_constraint(vol_limit=0.10):
    constraints = [
        weight_sum_constraint,
        {
            "type": "ineq",
            "fun": lambda w: vol_limit - portfolio_volatility(w),
        },  # vol ≤ limit
    ]
    result = minimize(
        fun=lambda w: -portfolio_return(
            w
        ),  # minimize negative return = maximize return
        x0=w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    return result


# ── Optimization 2: Min Volatility s.t. Return ≥ 4% ──────────────────────────
def opt2_min_vol_return_constraint(ret_floor=0.04):
    constraints = [
        weight_sum_constraint,
        {
            "type": "ineq",
            "fun": lambda w: portfolio_return(w) - ret_floor,
        },  # return ≥ floor
    ]
    result = minimize(
        fun=portfolio_volatility,
        x0=w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    return result


# ── Optimization 3: Max Sharpe Ratio ──────────────────────────────────────────
def opt3_max_sharpe(rf=0.0):
    constraints = [weight_sum_constraint]
    result = minimize(
        fun=lambda w: -portfolio_sharpe(w, rf),  # minimize negative Sharpe
        x0=w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    return result


# ── Run Optimizations ─────────────────────────────────────────────────────────
def print_result(title, result, label=""):
    w = result.x
    ret = portfolio_return(w)
    vol = portfolio_volatility(w)
    sr = ret / vol if vol > 0 else np.nan

    print(f"\n{'─' * 70}")
    print(f"  {title}")
    print(f"{'─' * 70}")
    print(f"  Status: {'Optimal' if result.success else 'FAILED — ' + result.message}")
    print(f"  Ann. Return:     {ret:.4f}  ({ret*100:.2f}%)")
    print(f"  Ann. Volatility: {vol:.4f}  ({vol*100:.2f}%)")
    print(f"  Sharpe Ratio:    {sr:.4f}")
    print(f"\n  Weights:")
    for i, a in enumerate(ASSETS):
        bar = "█" * int(w[i] * 40)
        print(f"    {a:25s}  {w[i]:.4f}  ({w[i]*100:5.1f}%)  {bar}")


print(f"\n{'=' * 70}")
print("OPTIMIZATION RESULTS")
print("=" * 70)

# 1. Max return, vol ≤ 10%
res1 = opt1_max_return_vol_constraint(vol_limit=0.10)
print_result("1. MAXIMIZE RETURN  s.t.  Annualized Vol ≤ 10%", res1)

# 2. Min vol, return ≥ 4%
res2 = opt2_min_vol_return_constraint(ret_floor=0.04)
print_result("2. MINIMIZE VOLATILITY  s.t.  Annualized Return ≥ 4%", res2)

# 3. Max Sharpe
res3 = opt3_max_sharpe(rf=0.0)
print_result("3. MAXIMIZE SHARPE RATIO", res3)

# ── Summary Table ─────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print("SUMMARY")
print("=" * 70)

summary_data = []
for label, res in [
    ("Max Return (vol≤10%)", res1),
    ("Min Vol (ret≥4%)", res2),
    ("Max Sharpe", res3),
]:
    w = res.x
    ret = portfolio_return(w)
    vol = portfolio_volatility(w)
    row = {"Objective": label, "Ann. Return": ret, "Ann. Vol": vol, "Sharpe": ret / vol}
    for i, a in enumerate(ASSETS):
        row[a] = w[i]
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data).set_index("Objective")
print(summary_df.to_string(float_format=lambda x: f"{x:.4f}"))

# Save results
summary_df.to_csv(os.path.join(os.path.dirname(__file__), "optimization_results.csv"))
print(f"\nSaved: optimization_results.csv")
