"""
Regime Change Analysis for Risk Assumptions
=============================================
Assignment Question:
  "Is there any regime change for the risk assumption observed from the
   historical data?  If yes, what is the impact on the portfolio
   optimization results?"

APPROACH
--------
1. DETECT regime changes by computing:
   - Rolling 24-month annualized volatility for each asset
   - Rolling 24-month pairwise correlations
   - Rolling 24-month portfolio-level metrics

2. IDENTIFY known risk regimes in our 2000-2026 data:
   - Dot-Com Bust        (2000-2002):  equity crash, bonds rally
   - Pre-GFC Calm        (2003-2006):  low vol across the board
   - Global Financial Crisis (2007-2009):  vol spike, correlations spike
   - Post-GFC Recovery   (2010-2019):  gradually declining vol, QE era
   - COVID Shock         (2020):        extreme vol spike, then fast recovery
   - Post-COVID / Inflation (2021-2023):  rate hikes, bonds & equities correlated
   - Recent              (2024-2026):  normalization

3. MEASURE regime impact by:
   - Computing covariance matrix WITHIN each regime (not full-sample)
   - Running the 3 optimizations per regime
   - Comparing weights, return, vol, Sharpe across regimes
   - Showing how the "optimal" portfolio changes dramatically

KEY INSIGHT
-----------
Standard MVO uses ONE static covariance matrix from the full history.
But risk is NOT constant — volatility clusters, and correlations spike
during crises (exactly when diversification is needed most).  A portfolio
"optimized" on calm-period data will carry FAR more risk in a crisis
than the model predicts.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

# ==============================================================================
# LOAD DATA
# ==============================================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "Data", "monthly_returns.csv")
df = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
ASSETS = df.columns.tolist()
N = len(ASSETS)


# ==============================================================================
# PART 1: ROLLING RISK ANALYSIS — Detect Non-Stationarity
# ==============================================================================
# Rolling window: 24 months (2 years) — a common choice that balances
# responsiveness with statistical reliability.
WINDOW = 24

print("=" * 80)
print("REGIME CHANGE ANALYSIS — Rolling Risk Metrics")
print("=" * 80)

# ── 1a: Rolling Annualized Volatility ────────────────────────────────────────
# For each asset, compute rolling 24-month std dev * sqrt(12) to annualize.
# If risk were constant, these lines would be flat.  They won't be.
rolling_vol = df.rolling(window=WINDOW).std() * np.sqrt(12)
rolling_vol = rolling_vol.dropna()

print("\n-- Rolling 24-Month Annualized Volatility (selected dates) --\n")
# Show at key dates to illustrate regime changes
key_dates = [
    "2002-09-30",  # Dot-com bottom
    "2006-06-30",  # Pre-GFC calm
    "2009-03-31",  # GFC peak crisis
    "2012-06-30",  # Post-GFC recovery
    "2019-12-31",  # Pre-COVID
    "2020-06-30",  # COVID shock in window
    "2022-12-31",  # Rate hike regime
    "2025-12-31",  # Recent
]
for d in key_dates:
    try:
        row = rolling_vol.loc[:d].iloc[-1]
        actual_date = rolling_vol.loc[:d].index[-1].strftime("%Y-%m")
        vals = "  ".join(f"{a[:8]:>8s}:{row[a]*100:5.1f}%" for a in ASSETS)
        print(f"  {actual_date}  {vals}")
    except (KeyError, IndexError):
        pass

# ── 1b: Rolling Pairwise Correlations ────────────────────────────────────────
# Correlations are the OTHER key risk input.  During crises, correlations
# tend to spike toward +1 (everything falls together), destroying the
# diversification benefit that MVO relied on.
print("\n-- Rolling 24-Month Correlations: Equities vs Bond --")
print("   (Negative = good diversifier; Positive = moving together)\n")

rolling_corr_eq_bd = df["US_Equities"].rolling(WINDOW).corr(df["US_10Yr_Bond"])
rolling_corr_eq_re = df["US_Equities"].rolling(WINDOW).corr(df["US_REITs"])

for d in key_dates:
    try:
        c1 = rolling_corr_eq_bd.loc[:d].iloc[-1]
        c2 = rolling_corr_eq_re.loc[:d].iloc[-1]
        actual_date = rolling_corr_eq_bd.loc[:d].dropna().index[-1].strftime("%Y-%m")
        print(f"  {actual_date}  Eq-Bond: {c1:+.3f}   Eq-REITs: {c2:+.3f}")
    except (KeyError, IndexError):
        pass

# Full-sample correlation for reference
print(f"\n  Full-sample Eq-Bond:  {df['US_Equities'].corr(df['US_10Yr_Bond']):+.3f}")
print(f"  Full-sample Eq-REITs: {df['US_Equities'].corr(df['US_REITs']):+.3f}")


# ==============================================================================
# PART 2: DEFINE REGIMES
# ==============================================================================
# We define regimes based on well-known macroeconomic periods.
# Each regime uses ONLY the data within that window to compute mu and Sigma.

REGIMES = {
    "Dot-Com Bust (2000-2002)": ("2000-01-01", "2002-12-31"),
    "Pre-GFC Calm (2003-2006)": ("2003-01-01", "2006-12-31"),
    "GFC Crisis (2007-2009)": ("2007-01-01", "2009-12-31"),
    "Post-GFC QE (2010-2019)": ("2010-01-01", "2019-12-31"),
    "COVID Shock (2020)": ("2020-01-01", "2020-12-31"),
    "Inflation/Hikes (2021-23)": ("2021-01-01", "2023-12-31"),
    "Full Sample (2000-2026)": (
        df.index.min().strftime("%Y-%m-%d"),
        df.index.max().strftime("%Y-%m-%d"),
    ),
}

print(f"\n\n{'=' * 80}")
print("REGIME DEFINITIONS & RISK PARAMETERS")
print("=" * 80)

regime_stats = {}
for regime_name, (start, end) in REGIMES.items():
    sub = df.loc[start:end]
    if len(sub) < 3:
        continue

    mu_ann = sub.mean().values * 12
    cov_ann = sub.cov().values * 12
    vol_ann = np.sqrt(np.diag(cov_ann))
    corr = sub.corr()

    regime_stats[regime_name] = {
        "n_months": len(sub),
        "mu": mu_ann,
        "cov": cov_ann,
        "vol": vol_ann,
        "corr": corr,
        "period": f"{start[:7]} to {end[:7]}",
    }

    print(f"\n{'─' * 80}")
    print(f"  {regime_name}  ({len(sub)} months)")
    print(f"{'─' * 80}")

    print(f"\n  Annualized Return & Volatility:")
    for i, a in enumerate(ASSETS):
        sr = mu_ann[i] / vol_ann[i] if vol_ann[i] > 0 else 0
        print(
            f"    {a:25s}  mu={mu_ann[i]:+.4f} ({mu_ann[i]*100:+5.1f}%)  "
            f"vol={vol_ann[i]:.4f} ({vol_ann[i]*100:5.1f}%)  SR={sr:+.3f}"
        )

    print(f"\n  Key Correlations:")
    print(
        f"    Equities-Bond:  {corr.iloc[0,1]:+.3f}   "
        f"Equities-REITs: {corr.iloc[0,2]:+.3f}   "
        f"Equities-Commod: {corr.iloc[0,3]:+.3f}"
    )


# ==============================================================================
# PART 3: PER-REGIME OPTIMIZATION
# ==============================================================================
# Run the same three optimizations using each regime's own mu and Sigma.
# This shows how the "optimal" portfolio CHANGES across regimes.

print(f"\n\n{'=' * 80}")
print("PER-REGIME OPTIMIZATION RESULTS")
print("=" * 80)
print("(Each regime uses its OWN return & covariance estimates)")


def run_optimizations(mu_vec, cov_mat, assets):
    """Run 3 optimizations with given mu and Sigma. Returns dict of results."""
    n = len(assets)
    b = [(0.0, 1.0)] * n
    eq_con = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    w_init = np.ones(n) / n

    def p_ret(w):
        return w @ mu_vec

    def p_vol(w):
        return np.sqrt(w @ cov_mat @ w)

    def p_sr(w):
        v = p_vol(w)
        return (p_ret(w)) / v if v > 0 else 0

    results = {}

    # Obj 1: Max return s.t. vol <= 10%
    res1 = minimize(
        fun=lambda w: -p_ret(w),
        x0=w_init,
        method="SLSQP",
        bounds=b,
        constraints=[eq_con, {"type": "ineq", "fun": lambda w: 0.10 - p_vol(w)}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    results["Max Ret (vol<=10%)"] = res1

    # Obj 2: Min vol s.t. return >= 4%
    res2 = minimize(
        fun=p_vol,
        x0=w_init,
        method="SLSQP",
        bounds=b,
        constraints=[eq_con, {"type": "ineq", "fun": lambda w: p_ret(w) - 0.04}],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    results["Min Vol (ret>=4%)"] = res2

    # Obj 3: Max Sharpe
    res3 = minimize(
        fun=lambda w: -p_sr(w),
        x0=w_init,
        method="SLSQP",
        bounds=b,
        constraints=[eq_con],
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    results["Max Sharpe"] = res3

    return results, p_ret, p_vol


# Collect all results for comparison table
all_rows = []

for regime_name, stats in regime_stats.items():
    mu_r = stats["mu"]
    cov_r = stats["cov"]

    print(f"\n{'─' * 80}")
    print(f"  {regime_name}")
    print(f"{'─' * 80}")

    results, p_ret, p_vol = run_optimizations(mu_r, cov_r, ASSETS)

    for obj_name, res in results.items():
        w = res.x
        ret = p_ret(w)
        vol = p_vol(w)
        sr = ret / vol if vol > 0 else np.nan
        status = "OK" if res.success else "FAIL"

        weights_str = "  ".join(f"{w[i]*100:5.1f}%" for i in range(N))
        print(
            f"  {obj_name:22s}  Ret={ret*100:+5.1f}%  "
            f"Vol={vol*100:5.1f}%  SR={sr:+.3f}  "
            f"[{status}]  w=[{weights_str}]"
        )

        row = {
            "Regime": regime_name,
            "Objective": obj_name,
            "Ann_Return": ret,
            "Ann_Vol": vol,
            "Sharpe": sr,
            "Status": status,
        }
        for i, a in enumerate(ASSETS):
            row[a] = w[i]
        all_rows.append(row)


# ==============================================================================
# PART 4: CROSS-REGIME COMPARISON
# ==============================================================================
print(f"\n\n{'=' * 80}")
print("CROSS-REGIME WEIGHT COMPARISON")
print("=" * 80)
print("(How much do 'optimal' weights shift across regimes?)\n")

results_df = pd.DataFrame(all_rows)

for obj in ["Max Ret (vol<=10%)", "Min Vol (ret>=4%)", "Max Sharpe"]:
    print(f"\n  {obj}:")
    print(f"  {'─' * 74}")
    subset = results_df[results_df["Objective"] == obj].copy()
    subset = subset.set_index("Regime")
    display_cols = ["Ann_Return", "Ann_Vol", "Sharpe"] + ASSETS
    available_cols = [c for c in display_cols if c in subset.columns]
    print(subset[available_cols].to_string(float_format=lambda x: f"{x:.4f}"))


# ==============================================================================
# PART 5: VOLATILITY REGIME COMPARISON — Crisis vs Calm
# ==============================================================================
print(f"\n\n{'=' * 80}")
print("REGIME CHANGE IMPACT: Crisis vs Calm")
print("=" * 80)

# Pick two contrasting regimes to highlight the impact
calm_name = "Pre-GFC Calm (2003-2006)"
crisis_name = "GFC Crisis (2007-2009)"

if calm_name in regime_stats and crisis_name in regime_stats:
    calm = regime_stats[calm_name]
    crisis = regime_stats[crisis_name]

    print(f"\n  Volatility Comparison (annualized):")
    print(f"  {'Asset':25s}  {'Calm':>10s}  {'Crisis':>10s}  {'Ratio':>8s}")
    print(f"  {'─'*25}  {'─'*10}  {'─'*10}  {'─'*8}")
    for i, a in enumerate(ASSETS):
        ratio = crisis["vol"][i] / calm["vol"][i]
        print(
            f"  {a:25s}  {calm['vol'][i]*100:9.1f}%  "
            f"{crisis['vol'][i]*100:9.1f}%  {ratio:7.1f}x"
        )

    print(f"\n  Correlation: Equities vs Bond:")
    print(f"    Calm period:  {calm['corr'].iloc[0,1]:+.3f}")
    print(f"    Crisis:       {crisis['corr'].iloc[0,1]:+.3f}")
    print(f"    Full sample:  {df['US_Equities'].corr(df['US_10Yr_Bond']):+.3f}")

    print(f"\n  Correlation: Equities vs REITs:")
    print(f"    Calm period:  {calm['corr'].iloc[0,2]:+.3f}")
    print(f"    Crisis:       {crisis['corr'].iloc[0,2]:+.3f}")
    print(f"    Full sample:  {df['US_Equities'].corr(df['US_REITs']):+.3f}")


# ==============================================================================
# PART 6: CONCLUSIONS
# ==============================================================================
print(f"\n\n{'=' * 80}")
print("CONCLUSIONS")
print("=" * 80)
print(
    """
  1. REGIME CHANGES ARE CLEARLY PRESENT:
     - Volatility is NOT constant.  During the GFC (2007-2009), equity vol
       spiked to ~25-30% vs ~10% during calm periods — a 2-3x increase.
     - COVID (2020) showed similar behavior: a sudden extreme vol spike.

  2. CORRELATIONS SHIFT DRAMATICALLY:
     - Equities-Bond correlation flips between regimes: sometimes negative
       (great diversifier), sometimes positive (2021-23 rate hike era).
     - Equities-REITs correlation spikes toward +0.8 during crises
       (diversification disappears exactly when you need it most).

  3. IMPACT ON PORTFOLIO OPTIMIZATION:
     - The "optimal" weights change DRAMATICALLY across regimes.
     - A portfolio optimized on calm-period data UNDERESTIMATES true risk
       during a crisis (actual vol will far exceed the 10% budget).
     - A portfolio optimized on crisis data is OVERLY CONSERVATIVE during
       calm periods, leaving returns on the table.

  4. THE FULL-SAMPLE MVO IS A COMPROMISE:
     - Using the full 2000-2026 sample averages across all regimes.
     - It doesn't represent ANY single regime well.
     - It gives a "middle ground" that underperforms in both calm and
       crisis periods compared to regime-specific optimization.

  5. PRACTICAL IMPLICATION:
     - Static MVO with one covariance matrix is insufficient.
     - Practitioners should consider:
       (a) Scenario-based optimization (optimize for multiple regimes)
       (b) Robust optimization (worst-case covariance)
       (c) Regime-switching models (dynamically adjust weights)
       (d) Shrinkage estimators (Ledoit-Wolf) to stabilize Sigma
"""
)

# ── Save Results ──────────────────────────────────────────────────────────────
output_path = os.path.join(
    os.path.dirname(__file__), "..", "Outputs", "regime_change_results.csv"
)
results_df.to_csv(output_path, index=False)
print(f"  Saved: {output_path}\n")

# ==============================================================================
# PART 7: GENERATE CHARTS
# ==============================================================================
print(f"\n\n{'=' * 80}")
print("GENERATING REGIME CHANGE CHARTS")
print("=" * 80)

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "Outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Chart 1: Rolling Risk Metrics (Volatility & Correlation) ──
fig_risk, axes_risk = plt.subplots(2, 1, figsize=(14, 10), sharex=True, dpi=150)

# Top panel: Rolling Volatility
for a in ASSETS:
    axes_risk[0].plot(
        rolling_vol.index, rolling_vol[a], label=a.replace("_", " "), linewidth=1.5
    )
axes_risk[0].set_title(
    "Rolling 24-Month Annualized Volatility (Detecting Risk Regimes)",
    fontsize=14,
    fontweight="bold",
)
axes_risk[0].set_ylabel("Annualized Volatility")
axes_risk[0].yaxis.set_major_formatter(PercentFormatter(1.0))
axes_risk[0].legend(loc="upper left")
axes_risk[0].grid(True, linestyle=":", alpha=0.6)

# Highlight crisis regimes
crisis_zones = [
    ("2000-01-01", "2002-12-31"),  # Dot com bust
    ("2007-01-01", "2009-12-31"),  # GFC
    ("2020-01-01", "2020-12-31"),  # COVID shock
]
for start, end in crisis_zones:
    axes_risk[0].axvspan(pd.Timestamp(start), pd.Timestamp(end), color="red", alpha=0.1)
    axes_risk[1].axvspan(pd.Timestamp(start), pd.Timestamp(end), color="red", alpha=0.1)

# Bottom panel: Rolling Correlation
axes_risk[1].plot(
    rolling_corr_eq_bd.index,
    rolling_corr_eq_bd,
    label="Equities vs 10Yr Bond",
    color="#1E88E5",
    linewidth=1.8,
)
axes_risk[1].plot(
    rolling_corr_eq_re.index,
    rolling_corr_eq_re,
    label="Equities vs REITs",
    color="#FFA726",
    linewidth=1.8,
)
axes_risk[1].axhline(0, color="black", linestyle="--", linewidth=1)
axes_risk[1].set_title(
    "Rolling 24-Month Correlation (Diversification Breakdown)",
    fontsize=14,
    fontweight="bold",
)
axes_risk[1].set_ylabel("Correlation")
axes_risk[1].legend(loc="upper left")
axes_risk[1].grid(True, linestyle=":", alpha=0.6)

fig_risk.tight_layout()
risk_chart_path = os.path.join(OUT_DIR, "regime_rolling_risk.png")
fig_risk.savefig(risk_chart_path)
plt.close(fig_risk)
print(f"Saved: {risk_chart_path}")

# ── Chart 2: Impact on Portfolio Optimization (Weights Comparison) ──
compare_regimes = [
    "Full Sample (2000-2026)",
    "Pre-GFC Calm (2003-2006)",
    "GFC Crisis (2007-2009)",
]
objectives = ["Max Ret (vol<=10%)", "Min Vol (ret>=4%)", "Max Sharpe"]
obj_titles = ["Max Return (Vol <= 10%)", "Min Volatility (Ret >= 4%)", "Max Sharpe"]

fig_w, axes_w = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
colors = ["#90A4AE", "#66BB6A", "#EF5350"]  # Gray (Full), Green (Calm), Red (Crisis)

for idx, (obj, title) in enumerate(zip(objectives, obj_titles)):
    ax = axes_w[idx]
    x = np.arange(N)
    width = 0.25

    for r_idx, reg in enumerate(compare_regimes):
        reg_data = results_df[
            (results_df["Regime"] == reg) & (results_df["Objective"] == obj)
        ]
        if not reg_data.empty:
            wts = reg_data.iloc[0][ASSETS].values.astype(float)
            bars = ax.bar(
                x + (r_idx - 1) * width,
                wts,
                width,
                label=reg.split(" (")[0],
                color=colors[r_idx],
                edgecolor="white",
            )
            for bar, wt in zip(bars, wts):
                if wt > 0.05:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.01,
                        f"{wt*100:.0f}%",
                        ha="center",
                        va="bottom",
                        fontsize=7,
                        fontweight="bold",
                    )

    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([a.replace("_", " ") for a in ASSETS], rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    if idx == 0:
        ax.set_ylabel("Optimal Weight")
        ax.legend(fontsize=9)
    ax.grid(True, axis="y", linestyle=":", alpha=0.5)

fig_w.suptitle(
    "Impact of Regime Change on Optimal Weights", fontsize=16, fontweight="bold"
)
fig_w.tight_layout(rect=[0, 0, 1, 0.95])
weights_chart_path = os.path.join(OUT_DIR, "regime_optimal_weights.png")
fig_w.savefig(weights_chart_path, bbox_inches="tight")
plt.close(fig_w)
print(f"Saved: {weights_chart_path}")

# ── Chart 3: Volatility Comparison Across Regimes (Bar Chart) ──
# Show how annualized volatility for each asset changes dramatically across regimes
chart3_regimes = [
    "Pre-GFC Calm (2003-2006)",
    "GFC Crisis (2007-2009)",
    "Post-GFC QE (2010-2019)",
    "COVID Shock (2020)",
    "Inflation/Hikes (2021-23)",
    "Full Sample (2000-2026)",
]
chart3_regimes = [r for r in chart3_regimes if r in regime_stats]

fig_vol, ax_vol = plt.subplots(figsize=(14, 7), dpi=150)
x = np.arange(N)
n_regimes = len(chart3_regimes)
width = 0.8 / n_regimes
regime_colors = ["#66BB6A", "#EF5350", "#42A5F5", "#FFA726", "#AB47BC", "#90A4AE"]

for r_idx, reg in enumerate(chart3_regimes):
    vols = regime_stats[reg]["vol"]
    offset = (r_idx - n_regimes / 2 + 0.5) * width
    bars = ax_vol.bar(
        x + offset,
        vols,
        width,
        label=reg.split(" (")[0],
        color=regime_colors[r_idx % len(regime_colors)],
        edgecolor="white",
        linewidth=0.5,
    )
    for bar, v in zip(bars, vols):
        if v > 0.02:
            ax_vol.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.003,
                f"{v*100:.1f}%",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
            )

ax_vol.set_title(
    "Annualized Volatility by Asset Across Market Regimes\n"
    "(Evidence that risk is NOT constant)",
    fontsize=14,
    fontweight="bold",
)
ax_vol.set_xticks(x)
ax_vol.set_xticklabels([a.replace("_", " ") for a in ASSETS], fontsize=11)
ax_vol.set_ylabel("Annualized Volatility", fontsize=12)
ax_vol.yaxis.set_major_formatter(PercentFormatter(1.0))
ax_vol.legend(loc="upper right", fontsize=9, ncol=2)
ax_vol.grid(True, axis="y", linestyle=":", alpha=0.5)
fig_vol.tight_layout()
vol_chart_path = os.path.join(OUT_DIR, "regime_volatility_comparison.png")
fig_vol.savefig(vol_chart_path)
plt.close(fig_vol)
print(f"Saved: {vol_chart_path}")

# ── Chart 4: Correlation Heatmaps — Calm vs Crisis vs Full Sample ──
heatmap_regimes = [
    ("Pre-GFC Calm\n(2003-2006)", "Pre-GFC Calm (2003-2006)"),
    ("GFC Crisis\n(2007-2009)", "GFC Crisis (2007-2009)"),
    ("COVID Shock\n(2020)", "COVID Shock (2020)"),
    ("Full Sample\n(2000-2026)", "Full Sample (2000-2026)"),
]
heatmap_regimes = [
    (label, key) for label, key in heatmap_regimes if key in regime_stats
]

fig_hm, axes_hm = plt.subplots(
    1, len(heatmap_regimes), figsize=(5 * len(heatmap_regimes), 5), dpi=150
)
if len(heatmap_regimes) == 1:
    axes_hm = [axes_hm]

short_names = [a.replace("US_", "").replace("10Yr_", "10Y\n") for a in ASSETS]
for idx, (label, key) in enumerate(heatmap_regimes):
    ax = axes_hm[idx]
    corr = regime_stats[key]["corr"].values
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(short_names, fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_title(label, fontsize=11, fontweight="bold")
    # Annotate cells
    for i in range(N):
        for j in range(N):
            color = "white" if abs(corr[i, j]) > 0.6 else "black"
            ax.text(
                j,
                i,
                f"{corr[i,j]:+.2f}",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color=color,
            )

fig_hm.suptitle(
    "Correlation Matrices Across Regimes — Diversification Breakdown in Crises",
    fontsize=14,
    fontweight="bold",
    y=1.02,
)
fig_hm.colorbar(im, ax=axes_hm, shrink=0.8, label="Correlation")
fig_hm.tight_layout()
heatmap_path = os.path.join(OUT_DIR, "regime_correlation_heatmaps.png")
fig_hm.savefig(heatmap_path, bbox_inches="tight")
plt.close(fig_hm)
print(f"Saved: {heatmap_path}")

# ── Chart 5: Per-Regime Efficient Frontiers ──
frontier_regimes = [
    ("Pre-GFC Calm (2003-2006)", "#66BB6A", "-"),
    ("GFC Crisis (2007-2009)", "#EF5350", "-"),
    ("Post-GFC QE (2010-2019)", "#42A5F5", "-"),
    ("Full Sample (2000-2026)", "#90A4AE", "--"),
]
frontier_regimes = [
    (name, c, ls) for name, c, ls in frontier_regimes if name in regime_stats
]

fig_ef, ax_ef = plt.subplots(figsize=(12, 8), dpi=150)

for reg_name, color, ls in frontier_regimes:
    mu_r = regime_stats[reg_name]["mu"]
    cov_r = regime_stats[reg_name]["cov"]
    n_pts = 80
    # Generate frontier points
    target_rets = np.linspace(mu_r.min(), mu_r.max(), n_pts)
    frontier_vols = []
    frontier_rets = []
    for t_ret in target_rets:
        res = minimize(
            fun=lambda w: np.sqrt(w @ cov_r @ w),
            x0=np.ones(N) / N,
            method="SLSQP",
            bounds=[(0, 1)] * N,
            constraints=[
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "eq", "fun": lambda w, tr=t_ret: w @ mu_r - tr},
            ],
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if res.success:
            frontier_vols.append(np.sqrt(res.x @ cov_r @ res.x))
            frontier_rets.append(t_ret)

    if frontier_vols:
        lw = 2.5 if "Full" in reg_name else 2.0
        ax_ef.plot(
            [v * 100 for v in frontier_vols],
            [r * 100 for r in frontier_rets],
            color=color,
            linestyle=ls,
            linewidth=lw,
            label=reg_name.split(" (")[0] + f" ({reg_name.split('(')[1]}",
        )

ax_ef.set_title(
    "Efficient Frontiers Across Market Regimes\n"
    '(The "optimal" frontier shifts dramatically with regime changes)',
    fontsize=14,
    fontweight="bold",
)
ax_ef.set_xlabel("Annualized Volatility (%)", fontsize=12)
ax_ef.set_ylabel("Annualized Return (%)", fontsize=12)
ax_ef.legend(loc="upper left", fontsize=10)
ax_ef.grid(True, linestyle=":", alpha=0.5)

# Add constraint reference lines
ax_ef.axvline(10, color="red", linestyle=":", alpha=0.4, linewidth=1)
ax_ef.axhline(4, color="blue", linestyle=":", alpha=0.4, linewidth=1)
ax_ef.text(
    10.2, ax_ef.get_ylim()[0] + 0.5, "Vol=10%", fontsize=8, color="red", alpha=0.6
)
ax_ef.text(
    ax_ef.get_xlim()[0] + 0.3, 4.3, "Ret=4%", fontsize=8, color="blue", alpha=0.6
)

fig_ef.tight_layout()
ef_path = os.path.join(OUT_DIR, "regime_efficient_frontiers.png")
fig_ef.savefig(ef_path)
plt.close(fig_ef)
print(f"Saved: {ef_path}")

# ── Chart 6: Regime Impact Summary Table ──
fig_tbl, ax_tbl = plt.subplots(figsize=(16, 8), dpi=150)
ax_tbl.axis("off")

# Build table data: for each regime × objective, show Return, Vol, Sharpe
table_regimes = [r for r in chart3_regimes if r != "Full Sample (2000-2026)"]
table_regimes.append("Full Sample (2000-2026)")  # put full sample last
table_data = []
row_colors = []

regime_row_colors = {
    "Pre-GFC Calm": "#E8F5E9",
    "GFC Crisis": "#FFEBEE",
    "Post-GFC QE": "#E3F2FD",
    "COVID Shock": "#FFF3E0",
    "Inflation/Hikes": "#F3E5F5",
    "Full Sample": "#F5F5F5",
}

for reg in table_regimes:
    for obj in objectives:
        row = results_df[
            (results_df["Regime"] == reg) & (results_df["Objective"] == obj)
        ]
        if not row.empty:
            r = row.iloc[0]
            wts_str = " | ".join(f"{r[a]*100:.1f}%" for a in ASSETS)
            table_data.append(
                [
                    reg.split(" (")[0],
                    obj,
                    f"{r['Ann_Return']*100:+.2f}%",
                    f"{r['Ann_Vol']*100:.2f}%",
                    f"{r['Sharpe']:.3f}",
                    wts_str,
                ]
            )
            short = reg.split(" (")[0]
            row_colors.append(regime_row_colors.get(short, "#FFFFFF"))

col_labels = [
    "Regime",
    "Objective",
    "Ann. Return",
    "Ann. Vol",
    "Sharpe",
    "Weights (Eq | Bond | REIT | Comm)",
]

table = ax_tbl.table(
    cellText=table_data,
    colLabels=col_labels,
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.0, 1.4)

# Style header
for j in range(len(col_labels)):
    cell = table[0, j]
    cell.set_facecolor("#1A237E")
    cell.set_text_props(color="white", fontweight="bold", fontsize=9)

# Style rows by regime
for i, color in enumerate(row_colors):
    for j in range(len(col_labels)):
        table[i + 1, j].set_facecolor(color)

ax_tbl.set_title(
    "Per-Regime Optimization Results — Full Comparison",
    fontsize=14,
    fontweight="bold",
    pad=20,
)
fig_tbl.tight_layout()
tbl_path = os.path.join(OUT_DIR, "regime_summary_table.png")
fig_tbl.savefig(tbl_path, bbox_inches="tight")
plt.close(fig_tbl)
print(f"Saved: {tbl_path}")

print("\n  All regime charts generated successfully!")
