"""
Equal Sharpe Ratio Analysis
============================
Assignment Question:
  "If assuming all assets have the same Sharpe Ratio, what are the optimization
   portfolios with these three objectives?  Will the optimization results depend
   on the Sharpe Ratio assumption?"

APPROACH
--------
1. Keep the covariance matrix (risk) UNCHANGED from historical data.
2. Replace expected returns so that every asset has the SAME Sharpe Ratio:
       mu_i = SR_common * sigma_i        (for each asset i)
   where sigma_i is the historical annualized volatility of asset i.
3. Re-run the three optimizations (max return, min vol, max Sharpe).
4. Repeat for several different SR_common values (0.3, 0.5, 0.7) to test
   whether the LEVEL of the common Sharpe Ratio changes the optimal weights.

KEY INSIGHT
-----------
When all assets share the same Sharpe Ratio:
  - The optimizer can NO LONGER distinguish assets by risk-adjusted attractiveness.
  - Higher-return assets are also proportionally higher-risk, so there is no
    "free lunch" from tilting toward any single asset.
  - The optimizations become PURELY driven by the covariance structure (correlations
    and volatilities), not by expected returns.
  - For Obj 1 (max return @ vol <= 10%): the solver picks the combination of
    assets that reaches exactly 10% vol, then return is mechanically determined
    by SR_common * 10% — weights depend on HOW to reach 10% vol most efficiently,
    which is a pure risk/correlation problem.
  - For Obj 2 (min vol @ ret >= 4%): similar — the minimum-variance portfolio
    that achieves the return floor, driven by correlations.
  - For Obj 3 (max Sharpe): ALL portfolios on the efficient frontier have the
    SAME Sharpe Ratio (it's a flat line on the SR axis), so the problem becomes
    degenerate — the minimum-variance portfolio is as good as any other.

DOES THE SR VALUE MATTER?
  - The WEIGHTS should be identical (or nearly so) regardless of whether
    SR_common = 0.3 or 0.7, because the weights are driven by the covariance
    structure, not the scale of returns.
  - Only the LEVEL of portfolio return/vol changes (proportionally).
  - EXCEPTION: Obj 2 (min vol @ ret >= 4%) may become infeasible if SR_common
    is too low (i.e. even 100% in the highest-return asset can't reach 4%).
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
# LOAD DATA (same as portfolio_optimization.py)
# ==============================================================================
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "Data", "monthly_returns.csv")
df = pd.read_csv(DATA_PATH, index_col="Date", parse_dates=True)
ASSETS = df.columns.tolist()
N = len(ASSETS)

# Historical covariance matrix (annualized) — this stays FIXED throughout
cov = df.cov().values * 12

# Historical volatilities (annualized) — used to construct equal-SR returns
sigma = np.sqrt(np.diag(cov))  # sigma_i for each asset

# Historical expected returns (for comparison)
mu_historical = df.mean().values * 12
sr_historical = mu_historical / sigma  # individual historical Sharpe Ratios


# ==============================================================================
# HELPER FUNCTIONS (parameterized by mu_vec so we can swap return assumptions)
# ==============================================================================
def port_return(w, mu_vec):
    """Portfolio return: R_p = w^T mu"""
    return w @ mu_vec


def port_vol(w):
    """Portfolio volatility: sigma_p = sqrt(w^T Sigma w)
    NOTE: volatility only depends on covariance, NOT on expected returns."""
    return np.sqrt(w @ cov @ w)


def port_sharpe(w, mu_vec, rf=0.0):
    """Sharpe Ratio: (R_p - Rf) / sigma_p"""
    return (port_return(w, mu_vec) - rf) / port_vol(w)


# ==============================================================================
# OPTIMIZATION FUNCTIONS (same three objectives, parameterized by mu_vec)
# ==============================================================================
bounds = [(0.0, 1.0)] * N
eq_constraint = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
w0 = np.ones(N) / N


def opt_max_return(mu_vec, vol_limit=0.10):
    """Obj 1: Maximize return s.t. volatility <= vol_limit."""
    constraints = [
        eq_constraint,
        {"type": "ineq", "fun": lambda w: vol_limit - port_vol(w)},
    ]
    return minimize(
        fun=lambda w: -port_return(w, mu_vec),
        x0=w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )


def opt_min_vol(mu_vec, ret_floor=0.04):
    """Obj 2: Minimize volatility s.t. return >= ret_floor."""
    constraints = [
        eq_constraint,
        {"type": "ineq", "fun": lambda w: port_return(w, mu_vec) - ret_floor},
    ]
    return minimize(
        fun=port_vol,
        x0=w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )


def opt_max_sharpe(mu_vec, rf=0.0):
    """Obj 3: Maximize Sharpe Ratio."""
    constraints = [eq_constraint]
    return minimize(
        fun=lambda w: -port_sharpe(w, mu_vec, rf),
        x0=w0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 1000},
    )


# ==============================================================================
# DISPLAY HELPERS
# ==============================================================================
def print_divider(char="=", width=80):
    print(char * width)


def print_weights(w, assets, prefix="    "):
    for i, a in enumerate(assets):
        bar = "#" * int(w[i] * 40)
        print(f"{prefix}{a:25s}  {w[i]:.4f}  ({w[i]*100:5.1f}%)  {bar}")


def run_and_display(label, mu_vec, assets):
    """Run all three optimizations with the given mu_vec and display results."""
    results = {}

    # Obj 1: Max Return s.t. Vol <= 10%
    res1 = opt_max_return(mu_vec, vol_limit=0.10)
    results["Max Return (vol<=10%)"] = res1

    # Obj 2: Min Vol s.t. Return >= 4%
    res2 = opt_min_vol(mu_vec, ret_floor=0.04)
    results["Min Vol (ret>=4%)"] = res2

    # Obj 3: Max Sharpe
    res3 = opt_max_sharpe(mu_vec)
    results["Max Sharpe"] = res3

    for obj_name, res in results.items():
        w = res.x
        ret = port_return(w, mu_vec)
        vol = port_vol(w)
        sr = ret / vol if vol > 0 else np.nan
        status = "Optimal" if res.success else f"FAILED - {res.message}"

        print(f"\n  {obj_name}")
        print(f"  {'─' * 60}")
        print(f"  Status:      {status}")
        print(f"  Ann. Return: {ret:.4f}  ({ret*100:.2f}%)")
        print(f"  Ann. Vol:    {vol:.4f}  ({vol*100:.2f}%)")
        print(f"  Sharpe:      {sr:.4f}")
        print(f"  Weights:")
        print_weights(w, assets)

    return results


# ==============================================================================
# MAIN ANALYSIS
# ==============================================================================
if __name__ == "__main__":

    # ── Part 1: Show Historical Sharpe Ratios ─────────────────────────────
    print_divider()
    print("EQUAL SHARPE RATIO ANALYSIS")
    print("Assignment: What happens when all assets have the same Sharpe Ratio?")
    print_divider()

    print("\n-- Historical Individual Sharpe Ratios (annualized, Rf=0) --\n")
    for i, a in enumerate(ASSETS):
        print(
            f"  {a:25s}  mu={mu_historical[i]:+.4f}  "
            f"sigma={sigma[i]:.4f}  SR={sr_historical[i]:.4f}"
        )
    print(f"\n  NOTE: Historical SRs are DIFFERENT across assets.")
    print(f"        This is what gives MVO leverage to tilt toward")
    print(f"        higher-SR assets.  What if they were all equal?\n")

    # ── Part 2: Run Historical Baseline ───────────────────────────────────
    print_divider("─")
    print("BASELINE: Historical Expected Returns (for comparison)")
    print_divider("─")
    baseline = run_and_display("Historical", mu_historical, ASSETS)

    # ── Part 3: Equal Sharpe Ratio — Multiple Levels ──────────────────────
    SR_VALUES = [0.3, 0.5, 0.7]

    # Store all results for the comparison table
    all_results = []

    for sr_val in SR_VALUES:
        print(f"\n\n")
        print_divider("=")
        print(f"EQUAL SHARPE RATIO:  SR = {sr_val}  (for ALL assets)")
        print_divider("=")

        # Construct adjusted expected returns:
        #   mu_i = SR_common * sigma_i
        # This ensures  SR_i = mu_i / sigma_i = SR_common  for every asset.
        mu_equal_sr = sr_val * sigma

        print(f"\n-- Adjusted Expected Returns (mu_i = {sr_val} * sigma_i) --\n")
        for i, a in enumerate(ASSETS):
            print(
                f"  {a:25s}  mu={mu_equal_sr[i]:+.4f} ({mu_equal_sr[i]*100:+.2f}%)  "
                f"sigma={sigma[i]:.4f} ({sigma[i]*100:.2f}%)  "
                f"SR={mu_equal_sr[i]/sigma[i]:.4f}"
            )

        results = run_and_display(f"SR={sr_val}", mu_equal_sr, ASSETS)

        # Collect for comparison
        for obj_name, res in results.items():
            w = res.x
            ret = port_return(w, mu_equal_sr)
            vol = port_vol(w)
            row = {
                "SR_assumption": sr_val,
                "Objective": obj_name,
                "Ann_Return": ret,
                "Ann_Vol": vol,
                "Sharpe": ret / vol if vol > 0 else np.nan,
                "Status": "Optimal" if res.success else "FAILED",
            }
            for i, a in enumerate(ASSETS):
                row[a] = w[i]
            all_results.append(row)

    # ── Part 4: Comparison Table ──────────────────────────────────────────
    print(f"\n\n")
    print_divider("=")
    print("COMPARISON: Do weights change when SR level changes?")
    print_divider("=")

    results_df = pd.DataFrame(all_results)
    results_df = results_df.set_index(["Objective", "SR_assumption"])

    # Show weights side-by-side for each objective
    for obj in ["Max Return (vol<=10%)", "Min Vol (ret>=4%)", "Max Sharpe"]:
        print(f"\n  {obj}:")
        print(f"  {'─' * 70}")
        subset = results_df.loc[obj]
        weight_cols = ASSETS
        display = subset[["Ann_Return", "Ann_Vol", "Sharpe"] + weight_cols]
        print(display.to_string(float_format=lambda x: f"{x:.4f}"))

    # ── Part 5: Conclusions ───────────────────────────────────────────────
    print(f"\n\n")
    print_divider("=")
    print("CONCLUSIONS")
    print_divider("=")
    print(
        """
  1. WEIGHTS are (nearly) IDENTICAL across different SR levels.
     - When all assets have the same Sharpe Ratio, the optimizer cannot
       exploit return differences.  Allocations are driven ENTIRELY by
       the covariance structure (correlations and volatilities).

  2. Only the SCALE of return/vol changes with SR level, not the weights.
     - Higher SR_common -> proportionally higher returns and same vol.

  3. Obj 3 (Max Sharpe) becomes DEGENERATE:
     - Every portfolio on the efficient frontier has the same Sharpe Ratio.
     - The solution collapses to the minimum-variance portfolio (since it
       achieves the same SR as any other portfolio, with lowest risk).

  4. Compared to HISTORICAL results, equal-SR produces VERY DIFFERENT weights:
     - Historical MVO heavily exploits return differences (e.g., overweighting
       high-SR assets like Equities, underweighting low-SR Commodities).
     - Equal-SR MVO focuses purely on risk reduction through diversification.

  5. ANSWER: Yes, optimization results DEPEND HEAVILY on return/Sharpe
     assumptions.  MVO is notoriously sensitive to expected return inputs.
     This is a well-known weakness called "estimation error amplification"
     (Michaud, 1989).  The equal-SR assumption reveals what the optimizer
     does when it has no return signal — it becomes a pure risk optimizer.
"""
    )

    # ── Save Results ──────────────────────────────────────────────────────
    output_path = os.path.join(
        os.path.dirname(__file__), "..", "Outputs", "equal_sharpe_results.csv"
    )
    results_df.to_csv(output_path)
    print(f"  Saved: {output_path}\n")

    # ==================================================================
    # CHART GENERATION — Equal Sharpe Ratio Analysis
    # ==================================================================
    OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "Outputs")

    # Use SR=0.5 as the primary equal-SR scenario for charts
    sr_primary = 0.5
    mu_eq = sr_primary * sigma

    # Re-run optimizations for the primary scenario
    eq_res1 = opt_max_return(mu_eq, vol_limit=0.10)
    eq_res2 = opt_min_vol(mu_eq, ret_floor=0.04)
    eq_res3 = opt_max_sharpe(mu_eq)

    # Also run historical baseline
    hist_res1 = opt_max_return(mu_historical, vol_limit=0.10)
    hist_res2 = opt_min_vol(mu_historical, ret_floor=0.04)
    hist_res3 = opt_max_sharpe(mu_historical)

    # ── Chart 1: Efficient Frontier Comparison (Historical vs Equal SR) ──
    print(f"\n{'=' * 70}")
    print("GENERATING EQUAL SHARPE RATIO CHARTS")
    print("=" * 70)

    w0_local = np.ones(N) / N

    # Generate frontier points for historical
    target_rets_hist = np.linspace(mu_historical.min(), mu_historical.max(), 80)
    hist_frontier = []
    for tr in target_rets_hist:
        cons = [
            eq_constraint,
            {"type": "eq", "fun": lambda w, tr=tr: port_return(w, mu_historical) - tr},
        ]
        r = minimize(
            port_vol, w0_local, method="SLSQP", bounds=bounds, constraints=cons
        )
        if r.success:
            hist_frontier.append((r.fun, tr))

    # Generate frontier points for equal SR
    target_rets_eq = np.linspace(mu_eq.min(), mu_eq.max(), 80)
    eq_frontier = []
    for tr in target_rets_eq:
        cons = [
            eq_constraint,
            {"type": "eq", "fun": lambda w, tr=tr: port_return(w, mu_eq) - tr},
        ]
        r = minimize(
            port_vol, w0_local, method="SLSQP", bounds=bounds, constraints=cons
        )
        if r.success:
            eq_frontier.append((r.fun, tr))

    fig, ax = plt.subplots(figsize=(12, 8), dpi=150)

    # Historical frontier
    if hist_frontier:
        hv, hr = zip(*hist_frontier)
        ax.plot(
            hv,
            hr,
            color="#999999",
            linewidth=2,
            linestyle="--",
            alpha=0.6,
            label="Historical Frontier",
            zorder=2,
        )

    # Equal SR frontier
    if eq_frontier:
        ev, er = zip(*eq_frontier)
        ax.plot(
            ev,
            er,
            color="#E53935",
            linewidth=2.5,
            label=f"Equal SR ({sr_primary}) Frontier",
            zorder=3,
        )

    # CML for equal SR through tangency
    eq_tangency_sr = port_sharpe(eq_res3.x, mu_eq)
    cml_v = np.linspace(0, 0.22, 100)
    cml_r = eq_tangency_sr * cml_v
    ax.plot(
        cml_v, cml_r, "k--", linewidth=1.2, alpha=0.5, label="CML (Equal SR)", zorder=1
    )

    # Individual assets (equal SR)
    ax.scatter(
        sigma,
        mu_eq,
        color="white",
        edgecolors="black",
        s=80,
        linewidths=1.5,
        label="Assets (Equal SR)",
        zorder=4,
    )
    for i, a in enumerate(ASSETS):
        ax.annotate(
            a.replace("_", " "),
            (sigma[i], mu_eq[i]),
            xytext=(8, -5),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color="#333333",
        )

    # Constraint lines
    ax.axvline(x=0.10, color="#E53935", linestyle=":", alpha=0.4, linewidth=1)
    ax.text(
        0.101, max(mu_eq) * 0.95, "Vol = 10%", fontsize=7, color="#E53935", alpha=0.6
    )
    ax.axhline(y=0.04, color="#1E88E5", linestyle=":", alpha=0.4, linewidth=1)
    ax.text(0.18, 0.041, "Ret = 4%", fontsize=7, color="#1E88E5", alpha=0.6)

    # Plot the 3 optimized portfolios (Equal SR) with info boxes
    eq_plot_data = [
        (eq_res1, "#E53935", "*", 250, "1. Max Return (Vol\u226410%)", (35, 25)),
        (eq_res2, "#1E88E5", "D", 180, "2. Min Vol (Ret\u22654%)", (-130, -50)),
        (eq_res3, "#FFA726", "P", 250, "3. Max Sharpe", (35, -45)),
    ]
    for res, color, marker, size, label, offset in eq_plot_data:
        w = res.x
        vol = port_vol(w)
        ret = port_return(w, mu_eq)
        sr = ret / vol if vol > 0 else 0
        ax.scatter(
            vol,
            ret,
            color=color,
            s=size,
            marker=marker,
            edgecolors="black",
            linewidths=1.5,
            label=label,
            zorder=7,
        )
        ax.annotate(
            f"Ret: {ret*100:.2f}%\nVol: {vol*100:.2f}%\nSR: {sr:.3f}",
            (vol, ret),
            xytext=offset,
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color=color,
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor="white",
                edgecolor=color,
                alpha=0.9,
                linewidth=1.5,
            ),
            arrowprops=dict(arrowstyle="->", color=color, lw=1.5),
            zorder=8,
        )

    ax.set_title(
        f"Efficient Frontier: Equal Sharpe Ratio (SR = {sr_primary}) vs Historical",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.set_xlabel("Annualized Volatility (Risk)", fontsize=12)
    ax.set_ylabel("Annualized Expected Return", fontsize=12)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend(loc="upper left", fontsize=8, frameon=True, fancybox=True, shadow=True)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "equal_sharpe_frontier.png"), bbox_inches="tight")
    plt.close(fig)
    print("Saved: equal_sharpe_frontier.png")

    # ── Chart 2: Weight Comparison (Historical vs Equal SR) all 3 objectives ──
    obj_names_display = [
        "Max Return\n(Vol\u226410%)",
        "Min Volatility\n(Ret\u22654%)",
        "Max Sharpe",
    ]
    hist_weights = [hist_res1.x, hist_res2.x, hist_res3.x]
    eq_weights_list = [eq_res1.x, eq_res2.x, eq_res3.x]
    panel_colors = ["#E53935", "#1E88E5", "#FFA726"]
    asset_labels = [a.replace("_", " ") for a in ASSETS]

    fig_w, axes_w = plt.subplots(1, 3, figsize=(18, 6), dpi=150)

    for idx, (ax_w, title, hw, ew) in enumerate(
        zip(axes_w, obj_names_display, hist_weights, eq_weights_list)
    ):
        x = np.arange(N)
        bw = 0.35
        bars_h = ax_w.bar(
            x - bw / 2, hw, bw, color="#90A4AE", label="Historical", edgecolor="white"
        )
        bars_e = ax_w.bar(
            x + bw / 2,
            ew,
            bw,
            color=panel_colors[idx],
            label=f"Equal SR ({sr_primary})",
            edgecolor="white",
        )

        for bar in bars_h:
            h = bar.get_height()
            if h > 0.03:
                ax_w.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + 0.01,
                    f"{h*100:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                    color="#666",
                )
        for bar in bars_e:
            h = bar.get_height()
            if h > 0.03:
                ax_w.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    h + 0.01,
                    f"{h*100:.0f}%",
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    fontweight="bold",
                )

        ax_w.set_title(title, fontsize=12, fontweight="bold")
        ax_w.set_xticks(x)
        ax_w.set_xticklabels(asset_labels, fontsize=8, rotation=15, ha="right")
        ax_w.set_ylabel("Weight" if idx == 0 else "")
        ax_w.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax_w.legend(fontsize=8)
        ax_w.grid(True, axis="y", linestyle=":", alpha=0.5)
        ax_w.set_ylim(0, 1.0)

    fig_w.suptitle(
        "Weight Allocation: Historical vs Equal Sharpe Ratio (All 3 Objectives)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig_w.tight_layout()
    fig_w.savefig(
        os.path.join(OUT_DIR, "equal_sharpe_weights.png"), bbox_inches="tight"
    )
    plt.close(fig_w)
    print("Saved: equal_sharpe_weights.png")

    # ── Chart 3: SR Sensitivity — Do weights change across SR levels? ──
    fig_s, axes_s = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    obj_keys = ["Max Return (vol<=10%)", "Min Vol (ret>=4%)", "Max Sharpe"]
    obj_display_s = [
        "Max Return (Vol\u226410%)",
        "Min Vol (Ret\u22654%)",
        "Max Sharpe",
    ]

    for idx, (ax_s, obj_key, obj_disp) in enumerate(
        zip(axes_s, obj_keys, obj_display_s)
    ):
        subset = results_df.loc[obj_key]
        x = np.arange(N)
        bw = 0.25
        for si, sr_val in enumerate(SR_VALUES):
            wts = subset.loc[sr_val][ASSETS].values.astype(float)
            bars = ax_s.bar(
                x + si * bw,
                wts,
                bw,
                label=f"SR = {sr_val}",
                alpha=0.85,
                edgecolor="white",
            )
            for bar, wt in zip(bars, wts):
                if wt > 0.05:
                    ax_s.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        bar.get_height() + 0.01,
                        f"{wt*100:.0f}%",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                        fontweight="bold",
                    )

        ax_s.set_title(obj_disp, fontsize=11, fontweight="bold")
        ax_s.set_xticks(x + bw)
        ax_s.set_xticklabels(asset_labels, fontsize=8, rotation=15, ha="right")
        ax_s.set_ylabel("Weight" if idx == 0 else "")
        ax_s.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax_s.legend(fontsize=8)
        ax_s.grid(True, axis="y", linestyle=":", alpha=0.5)
        ax_s.set_ylim(0, 1.0)

    fig_s.suptitle(
        "Do Weights Change When SR Level Changes? (SR = 0.3 vs 0.5 vs 0.7)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )
    fig_s.tight_layout()
    fig_s.savefig(
        os.path.join(OUT_DIR, "equal_sharpe_sensitivity.png"), bbox_inches="tight"
    )
    plt.close(fig_s)
    print("Saved: equal_sharpe_sensitivity.png")

    # ── Chart 4: Summary Comparison Table (rendered as image) ──
    fig_t, ax_t = plt.subplots(figsize=(16, 5), dpi=150)
    ax_t.axis("off")

    # Build table: Historical vs Equal SR (0.5) for each of 3 objectives
    table_data = []
    table_scenarios = [
        ("Historical", mu_historical, [hist_res1, hist_res2, hist_res3]),
        (f"Equal SR ({sr_primary})", mu_eq, [eq_res1, eq_res2, eq_res3]),
    ]
    obj_short = ["Max Return (Vol\u226410%)", "Min Vol (Ret\u22654%)", "Max Sharpe"]

    for scenario_name, mu_v, res_list in table_scenarios:
        for oi, (obj_label, res) in enumerate(zip(obj_short, res_list)):
            w = res.x
            ret = port_return(w, mu_v)
            vol = port_vol(w)
            sr = ret / vol if vol > 0 else 0
            wt_str = " | ".join([f"{w[j]*100:.1f}%" for j in range(N)])
            table_data.append(
                [
                    scenario_name,
                    obj_label,
                    f"{ret*100:+.2f}%",
                    f"{vol*100:.2f}%",
                    f"{sr:.3f}",
                    wt_str,
                ]
            )

    col_labels = [
        "Scenario",
        "Objective",
        "Ann. Return",
        "Ann. Vol",
        "Sharpe",
        "Weights (Eq|Bond|REIT|Comm)",
    ]
    tbl = ax_t.table(
        cellText=table_data, colLabels=col_labels, cellLoc="center", loc="center"
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.7)

    # Style header
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#2C3E50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")

    # Color rows by scenario
    for i in range(1, len(table_data) + 1):
        is_eq = i > 3  # rows 4-6 are equal SR
        bg = "#FFF8E1" if is_eq else "#F3F4F6"
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(bg)

    fig_t.suptitle(
        "Comparison: Historical MVO vs Equal Sharpe Ratio \u2014 All 3 Objectives",
        fontsize=13,
        fontweight="bold",
        y=0.98,
    )
    fig_t.tight_layout()
    fig_t.savefig(
        os.path.join(OUT_DIR, "equal_sharpe_comparison_table.png"), bbox_inches="tight"
    )
    plt.close(fig_t)
    print("Saved: equal_sharpe_comparison_table.png")

    print(f"\n{'=' * 70}")
    print("ALL PART 2 CHARTS GENERATED SUCCESSFULLY")
    print("=" * 70)
