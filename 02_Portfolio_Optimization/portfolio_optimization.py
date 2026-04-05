"""
Portfolio Optimization using Mean-Variance Optimization (MVO).

──────────────────────────────────────────────────────────────────────────────
BACKGROUND  (Markowitz, 1952)
──────────────────────────────────────────────────────────────────────────────
Mean-Variance Optimization finds the portfolio weights w = [w₁, w₂, …, wₙ]
that achieve the best trade-off between expected return and risk (volatility).

  Portfolio expected return :  R_p  = wᵀ μ          (dot product of weights and
                                                      expected return vector)
  Portfolio variance        :  σ²_p = wᵀ Σ w        (quadratic form using the
                                                      covariance matrix Σ)
  Portfolio volatility      :  σ_p  = √(wᵀ Σ w)
  Sharpe Ratio              :  SR   = (R_p − Rf) / σ_p

where μ is the vector of expected (annualized) returns, Σ is the annualized
covariance matrix, and Rf is the risk-free rate.

──────────────────────────────────────────────────────────────────────────────
THREE OPTIMIZATION OBJECTIVES  (required by the assignment)
──────────────────────────────────────────────────────────────────────────────
  1. Maximize return   subject to  annualized volatility ≤ 10%
  2. Minimize volatility  subject to  annualized return  ≥ 4%
  3. Maximize Sharpe Ratio  (best risk-adjusted return)

──────────────────────────────────────────────────────────────────────────────
CONSTRAINTS  (applied to all three problems)
──────────────────────────────────────────────────────────────────────────────
  • No short-selling:   wᵢ ≥ 0   for every asset i   (long-only portfolio)
  • Fully invested:     Σ wᵢ = 1                      (all capital deployed)

──────────────────────────────────────────────────────────────────────────────
SOLVER
──────────────────────────────────────────────────────────────────────────────
We use SciPy's `minimize` with the SLSQP (Sequential Least-Squares Quadratic
Programming) method. SLSQP handles equality + inequality constraints and
bound constraints, which is exactly what MVO requires.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# Use 'Agg' backend for non-interactive environments
import matplotlib

matplotlib.use("Agg")

# ── Load Data ─────────────────────────────────────────────────────────────────
# Read the CSV of monthly returns.  Each column is an asset, each row is a month.
# Example columns: "US Equities", "US 10Yr Treasury", "US REITs", "US Commodities"
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "Data", "monthly_returns.csv")
df = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
ASSETS = df.columns.tolist()  # list of asset names
N = len(ASSETS)  # number of assets (4 in our case)

# ── Annualized Inputs ─────────────────────────────────────────────────────────
# WHY ANNUALIZE?
#   The raw data is *monthly* returns.  We annualize so that the numbers are
#   easier to interpret and comparable to annual benchmarks (e.g. "10% vol").
#
# HOW?
#   • Expected return:  μ_annual = μ_monthly × 12
#       (assuming returns are additive / arithmetic mean)
#   • Covariance:       Σ_annual = Σ_monthly × 12
#       (variance scales linearly with time under i.i.d. assumption,
#        so multiplying the full covariance matrix by 12 annualizes
#        both variances on the diagonal and covariances off-diagonal)
#
# RESULT:
#   mu  → numpy array of shape (N,) with annualized expected returns
#   cov → numpy array of shape (N, N) with annualized covariance matrix
mu = df.mean().values * 12  # μ vector  (annualized expected returns)
cov = df.cov().values * 12  # Σ matrix  (annualized covariance matrix)

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
# These three functions compute the key portfolio metrics given a weight vector.
# The "@" operator in numpy performs matrix/vector multiplication (dot product).


def portfolio_return(w):
    """Annualized portfolio expected return:  R_p = wᵀ μ = Σ wᵢ μᵢ

    This is simply the weighted average of individual asset returns.
    Example: if w = [0.6, 0.4] and μ = [0.08, 0.03],
             R_p = 0.6×0.08 + 0.4×0.03 = 0.06 (6%)
    """
    return w @ mu  # numpy dot product: w₁μ₁ + w₂μ₂ + … + wₙμₙ


def portfolio_volatility(w):
    """Annualized portfolio volatility (standard deviation):
       σ_p = √(wᵀ Σ w)

    ── WHAT IS THE COVARIANCE MATRIX (Σ)? ──────────────────────────────────

    The covariance matrix is an N×N table that describes how every pair of
    assets moves together.  For our 4 assets it looks like:

                        Equities    Bond       REITs      Gold
        Equities     [  σ²_eq      cov(eq,bd)  cov(eq,re) cov(eq,au) ]
        Bond         [  cov(bd,eq) σ²_bd       cov(bd,re) cov(bd,au) ]
        REITs        [  cov(re,eq) cov(re,bd)  σ²_re      cov(re,au) ]
        Gold         [  cov(au,eq) cov(au,bd)  cov(au,re) σ²_au      ]

    • DIAGONAL (σ²): each asset's own variance (risk²).
      e.g. σ²_eq ≈ 0.0227  → σ_eq = √0.0227 ≈ 15.1% annual volatility

    • OFF-DIAGONAL (cov): how two assets co-move.
      - cov > 0 → they tend to move in the SAME direction
            e.g. cov(Equities, REITs) ≈ +0.0205  (both are equity-like,
                 they rise and fall together → less diversification benefit)
      - cov < 0 → they tend to move in OPPOSITE directions
            e.g. cov(Equities, Bond)  ≈ −0.0013  (when stocks drop, bonds
                 tend to rise → this REDUCES portfolio risk — diversification!)
      - cov ≈ 0 → nearly uncorrelated (independent movements)
            e.g. cov(Equities, Gold)  ≈ +0.0015  (weak relationship)

    ── HOW IS IT USED IN PORTFOLIO VARIANCE? ───────────────────────────────

    Portfolio variance expands to:

      σ²_p = wᵀ Σ w
           = Σᵢ Σⱼ  wᵢ wⱼ Σᵢⱼ

    Written out for 4 assets:

      σ²_p =  w_eq² × σ²_eq                          ← own risk of Equities
            + w_bd² × σ²_bd                            ← own risk of Bond
            + w_re² × σ²_re                            ← own risk of REITs
            + w_au² × σ²_au                            ← own risk of Gold
            + 2 × w_eq × w_bd × cov(eq,bd)            ← cross-term (negative!)
            + 2 × w_eq × w_re × cov(eq,re)            ← cross-term
            + 2 × w_eq × w_au × cov(eq,au)            ← cross-term
            + 2 × w_bd × w_re × cov(bd,re)            ← cross-term
            + 2 × w_bd × w_au × cov(bd,au)            ← cross-term
            + 2 × w_re × w_au × cov(re,au)            ← cross-term

    KEY INSIGHT:  The cross-terms with NEGATIVE covariance (like Equities vs
    Bond) SUBTRACT from total variance.  This is why mixing assets that move
    in opposite directions lowers overall portfolio risk — the mathematical
    basis of diversification.

    ── NUMPY COMPUTATION (what the code does) ──────────────────────────────

    Step by step:
      1.  cov @ w   → matrix-vector product → vector of length N
      2.  w @ (cov @ w)  → dot product → scalar = portfolio variance (σ²_p)
      3.  np.sqrt(...)   → square root → portfolio volatility (σ_p)
    """
    return np.sqrt(w @ cov @ w)


def portfolio_sharpe(w, rf=0.0):
    """Sharpe Ratio:  SR = (R_p − Rf) / σ_p

    Measures risk-adjusted return — how much excess return you earn per unit
    of risk (volatility).  A higher Sharpe Ratio is better.
      • rf = risk-free rate (default 0%, meaning we use raw return / vol)
      • Typical interpretation:  SR < 0.5 poor, 0.5–1.0 decent, >1.0 good
    """
    return (portfolio_return(w) - rf) / portfolio_volatility(w)


# ── Constraints & Bounds ──────────────────────────────────────────────────────
#
# BOUNDS (one per asset):
#   Each weight wᵢ must satisfy 0 ≤ wᵢ ≤ 1.
#   → (0.0, 1.0) means no short-selling (wᵢ ≥ 0) and no leverage (wᵢ ≤ 1).
#   We create N such bounds, one for each asset.
bounds = [(0.0, 1.0)] * N

# EQUALITY CONSTRAINT (fully invested):
#   Σ wᵢ = 1  →  expressed as:  Σ wᵢ − 1 = 0
#   SciPy equality constraints require a function that returns 0 at feasibility.
weight_sum_constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

# STARTING POINT:
#   SLSQP is an iterative solver — it needs an initial guess to start from.
#   We use equal weights (1/N each) as a reasonable, feasible starting point.
#   For N=4 assets, w0 = [0.25, 0.25, 0.25, 0.25].
w0 = np.ones(N) / N


# ── Optimization 1: Max Return s.t. Vol ≤ 10% ────────────────────────────────
#
# MATHEMATICAL FORMULATION:
#   maximize    wᵀ μ              (portfolio return)
#   subject to  √(wᵀ Σ w) ≤ 0.10 (annualized volatility ≤ 10%)
#               Σ wᵢ = 1          (fully invested)
#               wᵢ ≥ 0 ∀i        (no short-selling)
#
# INTUITION:  "Give me the highest return possible, but keep risk ≤ 10%."
#   The solver will push as much weight into high-return assets as possible
#   while staying within the volatility budget.
#
def opt1_max_return_vol_constraint(vol_limit=0.10):
    constraints = [
        weight_sum_constraint,  # Σ wᵢ = 1
        {
            # INEQUALITY CONSTRAINT:  SciPy requires  f(w) ≥ 0  for "ineq".
            # We want: σ_p ≤ vol_limit  →  rearrange to:  vol_limit − σ_p ≥ 0
            "type": "ineq",
            "fun": lambda w: vol_limit - portfolio_volatility(w),
        },
    ]
    result = minimize(
        # SciPy only MINIMIZES.  To MAXIMIZE return, we minimize the NEGATIVE:
        #   min  (−wᵀμ)   is equivalent to   max  (wᵀμ)
        fun=lambda w: -portfolio_return(w),
        x0=w0,  # start from equal weights
        method="SLSQP",  # Sequential Least-Squares Quadratic Programming
        bounds=bounds,  # 0 ≤ wᵢ ≤ 1
        constraints=constraints,
        options={
            "ftol": 1e-12,  # very tight convergence tolerance on objective
            "maxiter": 1000,  # max iterations (usually converges in <100)
        },
    )
    return result


# ── Optimization 2: Min Volatility s.t. Return ≥ 4% ──────────────────────────
#
# MATHEMATICAL FORMULATION:
#   minimize    √(wᵀ Σ w)        (portfolio volatility)
#   subject to  wᵀ μ ≥ 0.04      (annualized return ≥ 4%)
#               Σ wᵢ = 1          (fully invested)
#               wᵢ ≥ 0 ∀i        (no short-selling)
#
# INTUITION:  "Give me the safest (least volatile) portfolio that still
#   delivers at least 4% annual return."
#   The solver will diversify across low-correlation assets to minimize
#   volatility while meeting the return floor.
#
def opt2_min_vol_return_constraint(ret_floor=0.04):
    constraints = [
        weight_sum_constraint,  # Σ wᵢ = 1
        {
            # INEQUALITY CONSTRAINT:  SciPy "ineq" requires f(w) ≥ 0.
            # We want:  R_p ≥ ret_floor  →  R_p − ret_floor ≥ 0  ✓
            "type": "ineq",
            "fun": lambda w: portfolio_return(w) - ret_floor,
        },
    ]
    result = minimize(
        # Here the objective is ALREADY a minimization (minimize volatility),
        # so no negation trick is needed.
        fun=portfolio_volatility,
        x0=w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    return result


# ── Optimization 3: Max Sharpe Ratio ──────────────────────────────────────────
#
# MATHEMATICAL FORMULATION:
#   maximize    (wᵀ μ − Rf) / √(wᵀ Σ w)   (Sharpe Ratio)
#   subject to  Σ wᵢ = 1                    (fully invested)
#               wᵢ ≥ 0 ∀i                  (no short-selling)
#
# INTUITION:  "Find the portfolio with the best risk-adjusted return."
#   This is the tangency portfolio on the efficient frontier — the point where
#   a line from the risk-free rate is tangent to the frontier.  It represents
#   the optimal combination of risky assets regardless of risk preference.
#
# NOTE:  No additional return or volatility constraint here — the Sharpe Ratio
#   naturally balances return vs. risk.  The solver finds the "sweet spot".
#
def opt3_max_sharpe(rf=0.0):
    constraints = [weight_sum_constraint]  # only Σ wᵢ = 1
    result = minimize(
        # Again, SciPy only minimizes → minimize NEGATIVE Sharpe = maximize Sharpe
        fun=lambda w: -portfolio_sharpe(w, rf),
        x0=w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    return result


# ── Run Optimizations ─────────────────────────────────────────────────────────
def print_result(title, result, label=""):
    """Pretty-print the result of one optimization.

    Parameters
    ----------
    result : scipy.optimize.OptimizeResult
        .x       → optimal weight vector
        .success → True if solver converged to a solution
        .message → status message from the solver
    """
    w = result.x  # optimal weights (numpy array of length N)
    ret = portfolio_return(w)  # annualized return at optimal weights
    vol = portfolio_volatility(w)  # annualized volatility at optimal weights
    sr = ret / vol if vol > 0 else np.nan  # Sharpe Ratio

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
# Build a comparison table of all three optimization results side by side.
# Each row = one optimization objective, columns = return, vol, Sharpe, weights.
print(f"\n{'=' * 70}")
print("SUMMARY")
print("=" * 70)

summary_data = []
for label, res in [
    ("Max Return (vol≤10%)", res1),
    ("Min Vol (ret≥4%)", res2),
    ("Max Sharpe", res3),
]:
    w = res.x  # optimal weight vector
    ret = portfolio_return(w)  # annualized return
    vol = portfolio_volatility(w)  # annualized volatility
    row = {"Objective": label, "Ann. Return": ret, "Ann. Vol": vol, "Sharpe": ret / vol}
    for i, a in enumerate(ASSETS):  # add each asset's weight to the row
        row[a] = w[i]
    summary_data.append(row)

# Convert to DataFrame for clean tabular display
summary_df = pd.DataFrame(summary_data).set_index("Objective")
print(summary_df.to_string(float_format=lambda x: f"{x:.4f}"))

# Save to CSV so results can be referenced by other scripts (e.g. backtesting)
summary_df.to_csv(
    os.path.join(os.path.dirname(__file__), "..", "Outputs", "optimization_results.csv")
)
print(f"\nSaved: optimization_results.csv")

# ── Efficient Frontier Visualization ───────────────────────────────────────────
print(f"\n{'=' * 70}")
print("GENERATING EFFICIENT FRONTIER CHART")
print("=" * 70)

# 1. Generate Points for the Frontier
# Target returns ranging from the minimum possible volatility return to the max asset return
target_returns = np.linspace(mu.min(), mu.max(), 100)
frontier_vols = []

for tr in target_returns:
    cons = [
        weight_sum_constraint,
        {"type": "eq", "fun": lambda w, tr=tr: portfolio_return(w) - tr},
    ]
    res = minimize(
        portfolio_volatility, w0, method="SLSQP", bounds=bounds, constraints=cons
    )
    if res.success:
        frontier_vols.append(res.fun)
    else:
        frontier_vols.append(None)

# 1b. Generate random portfolios for Monte Carlo scatter
np.random.seed(42)
n_random = 5000
rand_rets, rand_vols, rand_sharpes = [], [], []
for _ in range(n_random):
    rw = np.random.dirichlet(np.ones(N))
    rr = portfolio_return(rw)
    rv = portfolio_volatility(rw)
    rand_rets.append(rr)
    rand_vols.append(rv)
    rand_sharpes.append(rr / rv if rv > 0 else 0)

# 2. Key Portfolios for Overlay
# Assignment portfolios
p1_w = res1.x
p1_metrics = [portfolio_volatility(p1_w), portfolio_return(p1_w)]

p2_w = res2.x
p2_metrics = [portfolio_volatility(p2_w), portfolio_return(p2_w)]

p3_w = res3.x
p3_metrics = [portfolio_volatility(p3_w), portfolio_return(p3_w)]

# Global Minimum Volatility
gmv_res = minimize(
    portfolio_volatility,
    w0,
    method="SLSQP",
    bounds=bounds,
    constraints=[weight_sum_constraint],
)
gmv_w = gmv_res.x
gmv_metrics = [portfolio_volatility(gmv_w), portfolio_return(gmv_w)]

# Equal Weight
ew_w = np.ones(N) / N
ew_metrics = [portfolio_volatility(ew_w), portfolio_return(ew_w)]

# 3. Plotting — Enhanced Efficient Frontier
fig, ax = plt.subplots(figsize=(12, 8), dpi=150)

# Monte Carlo feasible region (scatter of random portfolios)
sc = ax.scatter(
    rand_vols, rand_rets, c=rand_sharpes, cmap="viridis", alpha=0.3, s=8, zorder=0
)
cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.02)
cbar.set_label("Sharpe Ratio", fontsize=10)

# Plot the Frontier curve
valid_frontier = [
    (v, r) for v, r in zip(frontier_vols, target_returns) if v is not None
]
v_plot, r_plot = zip(*valid_frontier)
ax.plot(
    v_plot, r_plot, color="#E53935", linewidth=2.5, label="Efficient Frontier", zorder=3
)

# Capital Market Line (CML) from Rf=0 through tangency portfolio
tangency_sr = portfolio_sharpe(p3_w, rf=0.0)
cml_vols = np.linspace(0, max(v_plot) * 1.05, 100)
cml_rets = 0.0 + tangency_sr * cml_vols
ax.plot(
    cml_vols,
    cml_rets,
    "k--",
    linewidth=1.5,
    alpha=0.7,
    label="Capital Market Line (CML)",
    zorder=2,
)

# Individual asset points
asset_vols = [np.sqrt(cov[i, i]) for i in range(N)]
ax.scatter(
    asset_vols,
    mu,
    color="white",
    edgecolors="black",
    s=80,
    linewidths=1.5,
    label="Individual Assets",
    zorder=4,
)
for i, a in enumerate(ASSETS):
    ax.annotate(
        a.replace("_", " "),
        (asset_vols[i], mu[i]),
        xytext=(8, -5),
        textcoords="offset points",
        fontsize=9,
        fontweight="bold",
        color="#333333",
    )

# --- Constraint visualization lines ---
# Vertical line at 10% vol (constraint for Opt 1)
ax.axvline(x=0.10, color="#E53935", linestyle="--", alpha=0.5, linewidth=1.2, zorder=1)
ax.text(
    0.101,
    mu.max() * 0.98,
    "Vol = 10%\nconstraint",
    fontsize=7,
    color="#E53935",
    alpha=0.7,
    va="top",
)
# Horizontal line at 4% return (constraint for Opt 2)
ax.axhline(y=0.04, color="#1E88E5", linestyle="--", alpha=0.5, linewidth=1.2, zorder=1)
ax.text(
    max(v_plot) * 0.85,
    0.041,
    "Ret = 4% constraint",
    fontsize=7,
    color="#1E88E5",
    alpha=0.7,
    va="bottom",
)

# Reference portfolios (muted)
ref_markers = [
    (gmv_metrics, "#999999", "o", 60, "Global Min Vol"),
    (ew_metrics, "#999999", "X", 60, "Equal Weight (1/N)"),
]
for metrics, color, marker, size, label in ref_markers:
    ax.scatter(
        *metrics,
        color=color,
        s=size,
        marker=marker,
        edgecolors="gray",
        linewidths=0.5,
        label=label,
        zorder=4,
        alpha=0.6,
    )

# === THE THREE ASSIGNMENT PORTFOLIOS (emphasized) ===
assignment_portfolios = [
    (
        p1_metrics,
        "#E53935",
        "*",
        280,
        "1. Max Return\n   (Vol ≤ 10%)",
        f"Return: {p1_metrics[1]*100:.2f}%\nVol: {p1_metrics[0]*100:.2f}%\nSharpe: {p1_metrics[1]/p1_metrics[0]:.3f}",
        (40, 30),
    ),
    (
        p2_metrics,
        "#1E88E5",
        "D",
        200,
        "2. Min Volatility\n   (Ret ≥ 4%)",
        f"Return: {p2_metrics[1]*100:.2f}%\nVol: {p2_metrics[0]*100:.2f}%\nSharpe: {p2_metrics[1]/p2_metrics[0]:.3f}",
        (-120, -60),
    ),
    (
        p3_metrics,
        "#FFA726",
        "P",
        300,
        "3. Max Sharpe",
        f"Return: {p3_metrics[1]*100:.2f}%\nVol: {p3_metrics[0]*100:.2f}%\nSharpe: {p3_metrics[1]/p3_metrics[0]:.3f}",
        (40, -50),
    ),
]
for (
    metrics,
    color,
    marker,
    size,
    short_label,
    info_text,
    offset,
) in assignment_portfolios:
    ax.scatter(
        *metrics,
        color=color,
        s=size,
        marker=marker,
        edgecolors="black",
        linewidths=1.5,
        label=short_label,
        zorder=7,
    )
    ax.annotate(
        info_text,
        (metrics[0], metrics[1]),
        xytext=offset,
        textcoords="offset points",
        fontsize=8,
        fontweight="bold",
        color=color,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            edgecolor=color,
            alpha=0.9,
            linewidth=1.5,
        ),
        arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
        zorder=8,
    )

ax.set_title(
    "Efficient Frontier & Optimized Portfolios (Full Historical MVO)",
    fontsize=15,
    fontweight="bold",
    pad=15,
)
ax.set_xlabel("Annualized Volatility (Risk)", fontsize=12)
ax.set_ylabel("Annualized Expected Return", fontsize=12)
ax.xaxis.set_major_formatter(PercentFormatter(1.0))
ax.yaxis.set_major_formatter(PercentFormatter(1.0))
ax.grid(True, linestyle=":", alpha=0.4)
ax.legend(
    loc="upper left",
    fontsize=8,
    frameon=True,
    fancybox=True,
    shadow=True,
    framealpha=0.9,
)
ax.set_xlim(left=0)
ax.set_ylim(bottom=min(0, min(rand_rets) - 0.005))

fig.tight_layout()
out_img = os.path.join(
    os.path.dirname(__file__), "..", "Outputs", "efficient_frontier.png"
)
fig.savefig(out_img, bbox_inches="tight")
plt.close(fig)
print(f"Efficient Frontier chart saved to: {out_img}")

# ==============================================================================
# DEDICATED 3-PORTFOLIO COMPARISON (Assignment Requirement)
# ==============================================================================
print(f"\n{'=' * 70}")
print("GENERATING 3-PORTFOLIO COMPARISON CHART")
print("=" * 70)

assignment_names = [
    "Max Return (Vol≤10%)",
    "Min Volatility (Ret≥4%)",
    "Max Sharpe",
]
assignment_weights = [res1.x, res2.x, res3.x]
assignment_colors_3 = ["#E53935", "#1E88E5", "#FFA726"]

# Compute metrics for each
assign_records = []
assign_cum = {}
assign_dd = {}
for name, wt, clr in zip(assignment_names, assignment_weights, assignment_colors_3):
    pr = df.values @ wt
    ps = pd.Series(pr, index=df.index, name=name)
    cw = (1 + ps).cumprod()
    assign_cum[name] = cw
    rm = cw.cummax()
    dd = (cw / rm) - 1
    assign_dd[name] = dd
    ar = ps.mean() * 12
    av = ps.std() * np.sqrt(12)
    sr = ar / av if av > 0 else np.nan
    md = dd.min()
    assign_records.append(
        {
            "Portfolio": name,
            "Ann. Return": ar,
            "Ann. Vol": av,
            "Sharpe": sr,
            "Max Drawdown": md,
            "Calmar": ar / abs(md) if md != 0 else np.nan,
            "Final $1": cw.iloc[-1],
        }
    )
assign_df = pd.DataFrame(assign_records).set_index("Portfolio")

fig3, axes3 = plt.subplots(2, 2, figsize=(16, 11), dpi=150)

# Panel 1: Cumulative Wealth
ax_cw = axes3[0, 0]
for i, (name, cw) in enumerate(assign_cum.items()):
    ax_cw.plot(
        cw.index, cw.values, label=name, color=assignment_colors_3[i], linewidth=2.2
    )
ax_cw.set_title("Cumulative Wealth ($1 Invested)", fontsize=13, fontweight="bold")
ax_cw.set_ylabel("Portfolio Value ($)")
ax_cw.legend(fontsize=8, loc="upper left")
ax_cw.grid(True, linestyle=":", alpha=0.5)
ax_cw.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4)

# Panel 2: Drawdowns
ax_dd = axes3[0, 1]
for i, (name, dd) in enumerate(assign_dd.items()):
    ax_dd.fill_between(dd.index, dd.values, 0, alpha=0.2, color=assignment_colors_3[i])
    ax_dd.plot(
        dd.index, dd.values, color=assignment_colors_3[i], linewidth=1.5, label=name
    )
ax_dd.set_title("Drawdown Analysis", fontsize=13, fontweight="bold")
ax_dd.set_ylabel("Drawdown")
ax_dd.yaxis.set_major_formatter(PercentFormatter(1.0))
ax_dd.legend(fontsize=8, loc="lower left")
ax_dd.grid(True, linestyle=":", alpha=0.5)

# Panel 3: Metric bars (Sharpe, Calmar, Max DD side by side)
ax_bar = axes3[1, 0]
x3 = np.arange(3)
bw = 0.25
short_names = ["Max Return\n(Vol≤10%)", "Min Vol\n(Ret≥4%)", "Max Sharpe"]
b1 = ax_bar.bar(
    x3 - bw,
    assign_df["Sharpe"].values,
    bw,
    color="#42A5F5",
    label="Sharpe",
    edgecolor="white",
)
b2 = ax_bar.bar(
    x3,
    assign_df["Calmar"].values,
    bw,
    color="#66BB6A",
    label="Calmar",
    edgecolor="white",
)
b3 = ax_bar.bar(
    x3 + bw,
    assign_df["Max Drawdown"].abs().values,
    bw,
    color="#EF5350",
    label="|Max DD|",
    edgecolor="white",
)
for bars in [b1, b2, b3]:
    for bar in bars:
        h = bar.get_height()
        ax_bar.text(
            bar.get_x() + bar.get_width() / 2.0,
            h + 0.008,
            f"{h:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
ax_bar.set_title("Risk-Adjusted Metrics Comparison", fontsize=13, fontweight="bold")
ax_bar.set_xticks(x3)
ax_bar.set_xticklabels(short_names, fontsize=9)
ax_bar.legend(fontsize=9)
ax_bar.grid(True, axis="y", linestyle=":", alpha=0.5)

# Panel 4: Weight Allocation (horizontal stacked bar — one bar per portfolio)
ax_wt = axes3[1, 1]
asset_clr = ["#5C6BC0", "#26A69A", "#EF5350", "#FFA726"]
y_positions = np.arange(3)
left = np.zeros(3)

for aidx, (asset, aclr) in enumerate(zip(ASSETS, asset_clr)):
    wts = np.array([assignment_weights[p][aidx] for p in range(3)])
    bars = ax_wt.barh(
        y_positions,
        wts,
        left=left,
        height=0.6,
        label=asset.replace("_", " "),
        color=aclr,
        edgecolor="white",
        linewidth=0.5,
    )
    # Add percentage labels inside bars if weight > 5%
    for bar, w, l in zip(bars, wts, left):
        if w > 0.05:
            ax_wt.text(
                l + w / 2,
                bar.get_y() + bar.get_height() / 2,
                f"{w*100:.0f}%",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="white",
            )
    left += wts

ax_wt.set_title("Asset Weight Allocations", fontsize=13, fontweight="bold")
ax_wt.set_yticks(y_positions)
ax_wt.set_yticklabels(assignment_names, fontsize=10)
ax_wt.set_xlabel("Weight")
ax_wt.set_xlim(0, 1)
ax_wt.xaxis.set_major_formatter(PercentFormatter(1.0))
ax_wt.legend(fontsize=8, loc="lower right", ncol=2)
ax_wt.grid(True, axis="x", linestyle=":", alpha=0.5)

fig3.suptitle(
    "Three Assignment Portfolios — Performance Comparison",
    fontsize=16,
    fontweight="bold",
    y=1.01,
)
fig3.tight_layout()
fig3.savefig(
    os.path.join(
        os.path.dirname(__file__), "..", "Outputs", "three_portfolio_comparison.png"
    ),
    bbox_inches="tight",
)
plt.close(fig3)
print(f"Saved: three_portfolio_comparison.png")

# --- 3-Portfolio Summary Table (rendered as image) ---
fig3t, ax3t = plt.subplots(figsize=(16, 3), dpi=150)
ax3t.axis("off")

table3_data = []
for name in assign_df.index:
    r = assign_df.loc[name]
    wt = assignment_weights[assignment_names.index(name)]
    table3_data.append(
        [
            name,
            f"{wt[0]*100:.1f}%",  # US Equities
            f"{wt[1]*100:.1f}%",  # US 10Yr Bond
            f"{wt[2]*100:.1f}%",  # US REITs
            f"{wt[3]*100:.1f}%",  # US Commodities
            f"{r['Ann. Return']*100:+.2f}%",
            f"{r['Ann. Vol']*100:.2f}%",
            f"{r['Sharpe']:.3f}",
            f"{r['Max Drawdown']*100:.1f}%",
            f"{r['Calmar']:.3f}",
            f"${r['Final $1']:.2f}",
        ]
    )

col3 = [
    "Portfolio",
    "Equities",
    "Bonds",
    "REITs",
    "Commod.",
    "Ann. Return",
    "Ann. Vol",
    "Sharpe",
    "Max DD",
    "Calmar",
    "Final $1",
]
tbl3 = ax3t.table(cellText=table3_data, colLabels=col3, cellLoc="center", loc="center")
tbl3.auto_set_font_size(False)
tbl3.set_fontsize(9)
tbl3.scale(1.0, 1.8)

# Style
row_colors = ["#FDECEC", "#ECF2FD", "#FFF5E6"]  # red-ish, blue-ish, orange-ish tints
weight_col_bg = "#E8EAF6"  # light indigo tint for weight columns
for j in range(len(col3)):
    tbl3[0, j].set_facecolor("#2C3E50")
    tbl3[0, j].set_text_props(color="white", fontweight="bold")
for i in range(1, 4):
    for j in range(len(col3)):
        tbl3[i, j].set_facecolor(row_colors[i - 1])
        tbl3[i, j].set_edgecolor(assignment_colors_3[i - 1])
        tbl3[i, j].set_linewidth(1.5)
    # Highlight weight columns with a distinct tint
    for j in range(1, 5):
        tbl3[i, j].set_facecolor(weight_col_bg)

fig3t.suptitle(
    "Three Optimized Portfolios — Weights & Performance Summary",
    fontsize=13,
    fontweight="bold",
    y=0.98,
)
fig3t.tight_layout()
fig3t.savefig(
    os.path.join(
        os.path.dirname(__file__), "..", "Outputs", "three_portfolio_table.png"
    ),
    bbox_inches="tight",
)
plt.close(fig3t)
print(f"Saved: three_portfolio_table.png")

# ==============================================================================
# HISTORICAL PERFORMANCE ANALYSIS — Apply static MVO weights to actual returns
# ==============================================================================
# WHY?
#   The Efficient Frontier shows the THEORETICAL risk/return profile based on
#   the full-sample covariance and mean.  But in reality, an investor would have
#   held these weights through 2008, COVID, and other crises.  This section
#   simulates that experience and computes realized metrics (max drawdown, etc.)
# ==============================================================================

print(f"\n{'=' * 70}")
print("HISTORICAL PERFORMANCE ANALYSIS (Static Weights Applied to Real Returns)")
print("=" * 70)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "Outputs")

# Define all portfolios to analyse
portfolios = {
    "Equal Weight (1/N)": ew_w,
    "Max Return (Vol≤10%)": res1.x,
    "Min Volatility (Ret≥4%)": res2.x,
    "Max Sharpe": res3.x,
    "Global Min Volatility": gmv_w,
}

# --- Compute cumulative wealth and drawdowns for every portfolio ---
perf_records = []
cumulative_dict = {}
drawdown_dict = {}

for name, weights in portfolios.items():
    # Monthly portfolio return = weighted sum of asset returns at each time step
    port_rets = df.values @ weights  # shape: (T,)
    port_series = pd.Series(port_rets, index=df.index, name=name)

    # Cumulative wealth ($1 invested at inception)
    cum_wealth = (1 + port_series).cumprod()
    cumulative_dict[name] = cum_wealth

    # Drawdown = current wealth / peak wealth − 1  (always ≤ 0)
    running_max = cum_wealth.cummax()
    drawdown = (cum_wealth / running_max) - 1
    drawdown_dict[name] = drawdown

    # Annualized metrics
    ann_ret = port_series.mean() * 12
    ann_vol = port_series.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    max_dd = drawdown.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan
    final = cum_wealth.iloc[-1]

    perf_records.append(
        {
            "Portfolio": name,
            "Ann. Return": ann_ret,
            "Ann. Vol": ann_vol,
            "Sharpe": sharpe,
            "Max Drawdown": max_dd,
            "Calmar": calmar,
            "Final $1": final,
        }
    )

perf_df = pd.DataFrame(perf_records).set_index("Portfolio")
print(
    "\n"
    + perf_df.to_string(
        float_format=lambda x: f"{x:+.4f}" if abs(x) < 10 else f"${x:.2f}"
    )
)

# Save CSV
perf_df.to_csv(os.path.join(OUT_DIR, "mvo_historical_performance.csv"))
print(f"\nSaved: mvo_historical_performance.csv")

# --- Chart 1: Cumulative Wealth ---
colors = ["#7E57C2", "#E53935", "#1E88E5", "#FFA726", "#43A047"]
fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=120)

ax1 = axes[0, 0]
for i, (name, cum) in enumerate(cumulative_dict.items()):
    lw = 2.5 if "Max Sharpe" in name else 1.8
    ax1.plot(cum.index, cum.values, label=name, color=colors[i], linewidth=lw)
ax1.set_title("Cumulative Wealth ($1 Invested)", fontsize=13, fontweight="bold")
ax1.set_ylabel("Portfolio Value ($)")
ax1.legend(fontsize=8, loc="upper left")
ax1.grid(True, linestyle=":", alpha=0.5)
ax1.axhline(y=1.0, color="gray", linestyle="--", alpha=0.4)

# --- Chart 2: Drawdowns ---
ax2 = axes[0, 1]
for i, (name, dd) in enumerate(drawdown_dict.items()):
    ax2.fill_between(dd.index, dd.values, 0, alpha=0.25, color=colors[i])
    ax2.plot(dd.index, dd.values, color=colors[i], linewidth=1.2, label=name)
ax2.set_title("Drawdown Analysis", fontsize=13, fontweight="bold")
ax2.set_ylabel("Drawdown")
ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
ax2.legend(fontsize=8, loc="lower left")
ax2.grid(True, linestyle=":", alpha=0.5)

# --- Chart 3: Sharpe & Calmar Bar Chart ---
ax3 = axes[1, 0]
x_labels = [n.replace(" (", "\n(") for n in perf_df.index]
x_pos = np.arange(len(x_labels))
bar_width = 0.35

bars1 = ax3.bar(
    x_pos - bar_width / 2,
    perf_df["Sharpe"].values,
    bar_width,
    color="#42A5F5",
    label="Sharpe Ratio",
    edgecolor="white",
)
bars2 = ax3.bar(
    x_pos + bar_width / 2,
    perf_df["Calmar"].values,
    bar_width,
    color="#66BB6A",
    label="Calmar Ratio",
    edgecolor="white",
)

# Add value labels on bars
for bar in bars1:
    h = bar.get_height()
    ax3.text(
        bar.get_x() + bar.get_width() / 2.0,
        h + 0.01,
        f"{h:.2f}",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )
for bar in bars2:
    h = bar.get_height()
    ax3.text(
        bar.get_x() + bar.get_width() / 2.0,
        h + 0.01,
        f"{h:.2f}",
        ha="center",
        va="bottom",
        fontsize=8,
        fontweight="bold",
    )

ax3.set_title("Risk-Adjusted Performance Comparison", fontsize=13, fontweight="bold")
ax3.set_xticks(x_pos)
ax3.set_xticklabels(x_labels, fontsize=8)
ax3.legend(fontsize=9)
ax3.grid(True, axis="y", linestyle=":", alpha=0.5)

# --- Chart 4: Weight Allocation (Stacked Horizontal Bar) ---
ax4 = axes[1, 1]
weight_data = np.array([portfolios[name] for name in perf_df.index])
asset_colors = ["#5C6BC0", "#26A69A", "#EF5350", "#FFA726"]
left = np.zeros(len(perf_df))
for j, asset in enumerate(ASSETS):
    ax4.barh(
        x_pos,
        weight_data[:, j],
        left=left,
        height=0.6,
        label=asset.replace("_", " "),
        color=asset_colors[j],
        edgecolor="white",
    )
    # Add percentage labels inside bars if weight > 5%
    for k in range(len(perf_df)):
        wt = weight_data[k, j]
        if wt > 0.05:
            ax4.text(
                left[k] + wt / 2,
                x_pos[k],
                f"{wt*100:.0f}%",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
                color="white",
            )
    left += weight_data[:, j]

ax4.set_title("Portfolio Weight Allocations", fontsize=13, fontweight="bold")
ax4.set_yticks(x_pos)
ax4.set_yticklabels(x_labels, fontsize=8)
ax4.set_xlabel("Weight")
ax4.xaxis.set_major_formatter(PercentFormatter(1.0))
ax4.legend(fontsize=8, loc="lower right")
ax4.set_xlim(0, 1.0)

fig.suptitle(
    "MVO Portfolio Historical Performance Dashboard",
    fontsize=16,
    fontweight="bold",
    y=1.01,
)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "mvo_performance_dashboard.png"), bbox_inches="tight")
plt.close()
print(f"Saved: mvo_performance_dashboard.png")

# --- Performance Summary Table (rendered as image) ---
fig_table, ax_t = plt.subplots(figsize=(14, 4), dpi=150)
ax_t.axis("off")

# Format data for display
table_data = []
for name in perf_df.index:
    row = perf_df.loc[name]
    wt = portfolios[name]
    table_data.append(
        [
            name,
            f"{wt[0]*100:.1f}%",  # Equities
            f"{wt[1]*100:.1f}%",  # Bonds
            f"{wt[2]*100:.1f}%",  # REITs
            f"{wt[3]*100:.1f}%",  # Commodities
            f"{row['Ann. Return']*100:+.2f}%",
            f"{row['Ann. Vol']*100:.2f}%",
            f"{row['Sharpe']:.3f}",
            f"{row['Max Drawdown']*100:.1f}%",
            f"{row['Calmar']:.3f}",
            f"${row['Final $1']:.2f}",
        ]
    )

col_labels = [
    "Portfolio",
    "Equities",
    "Bonds",
    "REITs",
    "Commod.",
    "Ann. Return",
    "Ann. Vol",
    "Sharpe",
    "Max DD",
    "Calmar",
    "Final $1",
]
table = ax_t.table(
    cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center"
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.0, 1.6)

# Style header row
weight_col_bg = "#E8EAF6"  # light indigo for weight columns
for j in range(len(col_labels)):
    table[0, j].set_facecolor("#2C3E50")
    table[0, j].set_text_props(color="white", fontweight="bold")

# Alternate row colors
for i in range(1, len(table_data) + 1):
    bg = "#F8F9FA" if i % 2 == 0 else "#FFFFFF"
    for j in range(len(col_labels)):
        table[i, j].set_facecolor(bg)
    # Highlight weight columns
    for j in range(1, 5):
        table[i, j].set_facecolor(weight_col_bg)

# Highlight best Sharpe
best_sharpe_idx = perf_df["Sharpe"].argmax() + 1
for j in range(len(col_labels)):
    table[best_sharpe_idx, j].set_edgecolor("#FFA726")
    table[best_sharpe_idx, j].set_linewidth(2)

fig_table.suptitle(
    "MVO Optimized Portfolio — Historical Performance Summary",
    fontsize=13,
    fontweight="bold",
    y=0.98,
)
fig_table.tight_layout()
fig_table.savefig(
    os.path.join(OUT_DIR, "mvo_performance_table.png"), bbox_inches="tight"
)
plt.close(fig_table)
print(f"Saved: mvo_performance_table.png")

print(f"\n{'=' * 70}")
print("ALL PART 1 OUTPUTS GENERATED SUCCESSFULLY")
print("=" * 70)
